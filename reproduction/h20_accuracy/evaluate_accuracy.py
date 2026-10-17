"""Versioned, resumable author-protocol accuracy with per-request reuse evidence."""
import argparse
import asyncio
import collections
import contextlib
import fcntl
import hashlib
import io
import json
import math
import sys
import uuid
from pathlib import Path

import httpx
from openai import AsyncOpenAI
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'examples'))
from data import get_dataset
from metrics import get_qa_em,get_qa_f1,normalize,_rouge_scorer
from launch_component import PAIRS

CAPS={'hotpotqa':32768,'conversation':10240,'multinews':5120,'rulerqa':20480}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def atomic_json(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    tmp.replace(path)


def read_jsonl(path):
    with Path(path).open() as f:return [json.loads(line) for line in f if line.strip()]


def prepare(task):
    cache=HERE/'prepared'/task;cache.mkdir(parents=True,exist_ok=True)
    output=cache/'examples.jsonl';meta_path=cache/'manifest.json'
    data=ROOT/'examples/data/accuracy'/('rulerqa_100.jsonl' if task=='rulerqa' else task+'_1000.jsonl')
    config={'task':task,'data_sha256':sha(data),'loader_sha256':sha(ROOT/'examples/data.py'),
            'template_sha256':sha(ROOT/'examples/models/Qwen3-8B/tokenizer_config.json'),
            'max_length':CAPS[task],'num_max_examples':100 if task=='rulerqa' else 1000,
            'thinking':False}
    if meta_path.exists():
        old=json.loads(meta_path.read_text());assert old['config']==config
        assert sha(output)==old['sha256']
        return read_jsonl(output),old
    tok=AutoTokenizer.from_pretrained(ROOT/'examples/models/Qwen3-8B',local_files_only=True)
    logs=io.StringIO()
    with contextlib.redirect_stdout(logs):
        ds=get_dataset(task,str(data),tok,max_length=CAPS[task],num_max_examples=config['num_max_examples'])
    assert not logs.getvalue(),logs.getvalue()
    audited=json.loads((HERE/'checks'/f'{task}_selection.json').read_text())
    assert len(audited)==len(ds)
    rows=[]
    for i,x in enumerate(ds):
        row={'index':i,'sender_messages':x['sender_messages'],'receiver_messages':x['receiver_messages'],'gold':x['ans']}
        row['example_sha256']=hashlib.sha256(json.dumps({'sender':x['sender_messages'],'receiver':x['receiver_messages'],'gold':x['ans']},sort_keys=True).encode()).hexdigest()
        assert row['example_sha256']==audited[i]['example_sha256']
        for side in ['sender','receiver']:
            ids=tok.apply_chat_template(row[side+'_messages'],tokenize=True,add_generation_prompt=True,enable_thinking=False)
            row[side+'_tokens']=len(ids)
            row[side+'_ids_sha256']=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            assert row[side+'_ids_sha256']==audited[i][side+'_ids_sha256']
        row['shared_blocks_64']=audited[i]['shared_blocks_64']
        rows.append(row)
    tmp=output.with_suffix('.tmp')
    with tmp.open('w') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    tmp.replace(output)
    meta={'config':config,'n':len(rows),'sha256':sha(output)}
    atomic_json(meta_path,meta)
    return rows,meta


def scores(task,text,gold):
    if task in ['hotpotqa','rulerqa']:
        result={'em':get_qa_em(text,gold),'f1':get_qa_f1(text,gold)}
        if task=='hotpotqa':
            p,g=normalize(text),normalize(gold)
            result['official_answer_f1']=0. if p!=g and (p in {'yes','no','noanswer'} or g in {'yes','no','noanswer'}) else result['f1']
        return result
    # Same stemmed ROUGE-L as author code, but propagate metric exceptions.
    value=_rouge_scorer.score(gold.strip(),text.strip())['rougeL'].fmeasure if text.strip() and gold.strip() else 0.
    return {'rougeL':value}


def summarize(preds):
    assert preds
    fields=preds[0]['scores']
    return {'n':len(preds),**{k:math.fsum(x['scores'][k] for x in preds)/len(preds) for k in fields},
            'length_stops':sum(x['finish_reason']=='length' for x in preds),
            'with_reuse':sum(x['reuse']['reused_layers']>0 for x in preds),
            'zero_shared_blocks':sum(x['shared_blocks_64']==0 for x in preds)}


def fingerprint(pair,task):
    s,t,*_=PAIRS[pair]
    paths=list((ROOT/'mapflow').rglob('*.py'))+[Path(__file__),HERE/'launch_component.py',
        ROOT/'examples/data.py',ROOT/'examples/metrics.py']
    paths+=list((ROOT/'examples/weights'/f'{s}_to_{t}'/task).glob('translator_layer_*.pt'))
    for model in [s,t]:
        paths += [ROOT/'examples/models'/model/name for name in
                  ['config.json','generation_config.json','tokenizer_config.json','model.safetensors.index.json']]
    import vllm
    runtime=Path(vllm.__file__).parent
    paths += [runtime/name for name in ['v1/attention/backends/bsr_attn.py','v1/attention/backends/flash_attn.py',
        'model_executor/layers/attention/attention.py','v1/core/sched/scheduler.py','v1/worker/gpu_model_runner.py']]
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}


def audit(out):
    out=Path(out);p=json.loads((out/'protocol.json').read_text())
    rows,meta=prepare(p['task']);rows=rows[:p['limit']] if p['limit'] else rows
    assert meta==p['data']
    preds=read_jsonl(out/'predictions.jsonl')
    assert len(preds)==len(rows) and {x['index'] for x in preds}==set(range(len(rows)))
    for x in preds:
        r=rows[x['index']]
        assert x['example_sha256']==r['example_sha256'] and x['gold']==r['gold']
        assert x['scores']==scores(p['task'],x['prediction'],r['gold'])
        assert x['response']['choices'][0]['message']['content']==x['prediction']
        assert x['response']['choices'][0]['finish_reason']==x['finish_reason']
        assert x['response']['usage']['prompt_tokens']==r['receiver_tokens']
        events=x['reuse']['events']
        reused={e['layer'] for e in events if e['reused'] and e['kernel_completed'] and e['nnz']>0}
        assert x['reuse']['reused_layers']==len(reused)
        if p['mode']=='mapflow':
            assert x['source_response']['usage']['prompt_tokens']==r['sender_tokens']
            assert events and all(any(x['request_id'] in rid for rid,_ in e['requests']) for e in events)
            if r['shared_blocks_64']:
                assert reused==set(range(3,p['expected_reuse_layers']+3)),x['index']
        else:
            assert not events and x['source_response'] is None
    result={'complete':True,'protocol_sha256':sha(out/'protocol.json'),
            'predictions_sha256':sha(out/'predictions.jsonl'),**summarize(preds)}
    atomic_json(out/'results.json',result)
    atomic_json(out/'audit.json',{'passed':True,'n':len(preds),'checks':['exact sample coverage','data manifest',
        'reference identity','prompt token counts','prediction rescore','per-request reuse layers'],'result':result})
    return result


async def run(a):
    rows,meta=prepare(a.task);rows=rows[:a.limit] if a.limit else rows
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    s,t,sl,tl,*_=PAIRS[a.pair]
    generation={'temperature':.6,'top_p':.95,'top_k':20,'seed_base':42,
                'max_tokens':32 if a.task in ['hotpotqa','rulerqa'] else 1024,'enable_thinking':False}
    protocol={'pair':a.pair,'task':a.task,'mode':a.mode,'data':meta,'limit':a.limit,
              'generation':generation,'source_fingerprint':fingerprint(a.pair,a.task),
              'expected_reuse_layers':tl-4,'sender_wait_seconds':6. if a.task=='rulerqa' else .5,
              'sampling_note':'Author model defaults, explicitly set; added per-item deterministic seeds.',
              'native_note':'Native uses identical receiver prompt and sampling; sender request unnecessary.',
              'stack_manifests':{role:json.loads((a.stack_dir/f'{role}.launch.json').read_text()) for role in
                  (['server','sender','receiver'] if a.mode=='mapflow' else ['native'])}}
    # Runtime PID is not an experiment parameter; retain launcher content minus PID.
    for v in protocol['stack_manifests'].values():v.pop('pid',None)
    with (out/'job.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pp=out/'protocol.json'
        if pp.exists():assert json.loads(pp.read_text())==protocol,'incompatible protocol; use a new output directory'
        else:atomic_json(pp,protocol)
        predpath=out/'predictions.jsonl';done=read_jsonl(predpath) if predpath.exists() else []
        assert len({x['index'] for x in done})==len(done)
        finished={x['index'] for x in done};assert finished<=set(range(len(rows)))
        for x in done:assert x['example_sha256']==rows[x['index']]['example_sha256']
        async with AsyncOpenAI(base_url=f'http://127.0.0.1:{a.base_port+1}/v1',api_key='EMPTY',max_retries=0,timeout=600,
                   http_client=httpx.AsyncClient(trust_env=False,timeout=600)) as sender, \
                   AsyncOpenAI(base_url=f'http://127.0.0.1:{a.base_port+(2 if a.mode=="mapflow" else 3)}/v1',api_key='EMPTY',max_retries=0,timeout=600,
                   http_client=httpx.AsyncClient(trust_env=False,timeout=600)) as receiver:
            await receiver.models.list()
            if a.mode=='mapflow':await sender.models.list()
            with predpath.open('a',buffering=1) as f:
                for row in rows:
                    i=row['index']
                    if i in finished:continue
                    request_id=f'mapflow-{a.pair}-{a.task}-{a.mode}-{i}-{uuid.uuid4().hex[:12]}'
                    extra={'chat_template_kwargs':{'enable_thinking':False},'top_k':20}
                    source_response=None
                    if a.mode=='mapflow':
                        source_response=await sender.chat.completions.create(model=s,messages=row['sender_messages'],max_tokens=1,
                            temperature=.6,top_p=.95,seed=42+i,extra_body={**extra,'request_id':request_id+'-source'})
                        assert source_response.usage.prompt_tokens==row['sender_tokens']
                        await asyncio.sleep(protocol['sender_wait_seconds'])
                    eventpath=a.stack_dir/'reuse_events.jsonl'
                    offset=eventpath.stat().st_size if eventpath.exists() else 0
                    response=await receiver.chat.completions.create(model=t,messages=row['receiver_messages'],
                        max_tokens=generation['max_tokens'],temperature=.6,top_p=.95,seed=42+i,
                        extra_body={**extra,'request_id':request_id})
                    assert response.usage.prompt_tokens==row['receiver_tokens']
                    assert len(response.choices)==1 and response.choices[0].finish_reason in ['stop','length']
                    prediction=response.choices[0].message.content
                    assert isinstance(prediction,str)
                    events=[]
                    if a.mode=='mapflow':
                        with eventpath.open() as ef:
                            ef.seek(offset)
                            events=[json.loads(line) for line in ef if line.strip()]
                        events=[e for e in events if any(request_id in rid for rid,_ in e['requests'])]
                        assert events, 'missing per-request prefill evidence'
                    reuse_layers={e['layer'] for e in events if e['reused'] and e['kernel_completed'] and e['nnz']>0}
                    if a.mode=='mapflow' and row['shared_blocks_64']:
                        assert reuse_layers==set(range(3,tl-1)),f'Incomplete reuse for sample {i}: {reuse_layers}'
                    record={'index':i,'example_sha256':row['example_sha256'],'gold':row['gold'],
                        'prediction':prediction,'scores':scores(a.task,prediction,row['gold']),
                        'shared_blocks_64':row['shared_blocks_64'],'request_id':request_id,
                        'finish_reason':response.choices[0].finish_reason,'response':response.model_dump(),
                        'source_response':source_response.model_dump() if source_response else None,
                        'reuse':{'reused_layers':len(reuse_layers),'events':events}}
                    f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');done.append(record)
                    print(json.dumps({'done':len(done),'total':len(rows),**summarize(done)}),flush=True)
    print(json.dumps(audit(out),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',choices=PAIRS,required=True)
    p.add_argument('--task',choices=CAPS,required=True);p.add_argument('--mode',choices=['native','mapflow'],required=True)
    p.add_argument('--stack-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--base-port',type=int,default=7000);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--audit-only',action='store_true');a=p.parse_args()
    if a.audit_only:print(json.dumps(audit(a.output),indent=2))
    else:asyncio.run(run(a))
