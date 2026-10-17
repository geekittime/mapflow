"""Audit exact author-loader selections and released weights before GPU runs."""
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'examples'))
from data import get_dataset


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    out = Path(__file__).parent / 'checks'
    tok = AutoTokenizer.from_pretrained('/home/tiger/models/Qwen3-8B', local_files_only=True)
    tokenizer_checks = {}
    for model in ['Qwen3-1.7B', 'Qwen3-14B']:
        other = AutoTokenizer.from_pretrained('/home/tiger/models/' + model, local_files_only=True)
        tokenizer_checks[model] = {'vocab_equal': tok.get_vocab() == other.get_vocab(),
                                  'template_equal': tok.chat_template == other.chat_template}
        assert all(tokenizer_checks[model].values())
    report = {'tokenizers': tokenizer_checks, 'datasets': {}, 'weights': {}}
    for task, cap in [('hotpotqa', 32768), ('conversation', 10240), ('multinews', 5120), ('rulerqa', 20480)]:
        name = 'rulerqa_100.jsonl' if task == 'rulerqa' else task + '_1000.jsonl'
        path = ROOT / 'examples/data/accuracy' / name
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            ds = get_dataset(task, str(path), tok, max_length=cap, num_max_examples=100 if task == 'rulerqa' else 1000)
        assert not log.getvalue(), log.getvalue()
        entries = []
        for i, r in enumerate(ds):
            sender = tok.apply_chat_template(r['sender_messages'], tokenize=True, add_generation_prompt=True, enable_thinking=False)
            receiver = tok.apply_chat_template(r['receiver_messages'], tokenize=True, add_generation_prompt=True, enable_thinking=False)
            common = 0
            for a, b in zip(sender, receiver):
                if a != b:
                    break
                common += 1
            entry = {'index': i, 'sender_tokens': len(sender), 'receiver_tokens': len(receiver),
                     'shared_prefix_tokens': common, 'shared_blocks_64': common // 64,
                     'sender_ids_sha256': hashlib.sha256(json.dumps(sender).encode()).hexdigest(),
                     'receiver_ids_sha256': hashlib.sha256(json.dumps(receiver).encode()).hexdigest(),
                     'example_sha256': hashlib.sha256(json.dumps({'sender': r['sender_messages'], 'receiver': r['receiver_messages'], 'gold': r['ans']}, sort_keys=True).encode()).hexdigest()}
            entries.append(entry)
        assert entries, task
        report['datasets'][task] = {'file': str(path.relative_to(ROOT)), 'sha256': digest(path),
            'raw_records': sum(1 for _ in path.open()), 'selected': len(entries), 'length_cap_author_loader': cap,
            'receiver_tokens_min': min(x['receiver_tokens'] for x in entries),
            'receiver_tokens_max': max(x['receiver_tokens'] for x in entries),
            'zero_shared_blocks': sum(x['shared_blocks_64'] == 0 for x in entries)}
        (out / (task + '_selection.json')).write_text(json.dumps(entries, indent=2) + '\n')
        print(task, report['datasets'][task], flush=True)
    for pair, shape, layers in [('Qwen3-8B_to_Qwen3-1.7B', (32,16),28), ('Qwen3-14B_to_Qwen3-8B',(40,32),36)]:
        for task in report['datasets']:
            d = ROOT / 'examples/weights' / pair / task
            files = sorted(d.glob('translator_layer_*.pt'))
            assert len(files) == layers - 3
            assert {int(p.stem.rsplit('_', 1)[1]) for p in files} == set(range(3, layers))
            details = []
            for p in files:
                checkpoint = torch.load(p, map_location='cpu', weights_only=True)
                # Author saves a state_dict; runtime explicitly reads ['weight'].
                assert isinstance(checkpoint, dict) and 'weight' in checkpoint, p
                w = checkpoint['weight']
                assert tuple(w.shape) == shape and torch.isfinite(w).all(), p
                z = w.float().softmax(0)
                torch.testing.assert_close(z.sum(0), torch.ones(shape[1]), atol=1e-6, rtol=1e-6)
                upstream = Path('/home/tiger/mapflow-upstream') / p.relative_to(ROOT)
                assert digest(p) == digest(upstream), f'weight differs from author: {p}'
                details.append({'file': p.name, 'sha256': digest(p), 'shape': list(w.shape),
                                'author_hash_match': True})
            report['weights'][pair + '/' + task] = details
    (out / 'inputs.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
