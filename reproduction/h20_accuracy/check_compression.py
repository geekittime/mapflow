"""Compare actual author compression kernels with dense causal GQA attention."""
import importlib.util
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('compression_under_test',ROOT/'mapflow/kernel/compress_page_triton.py')
kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernel)


def main():
    torch.manual_seed(17)
    report={'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),'cases':[]}
    for tokens,q_tokens in [(256,256),(256,128),(1024,256)]:
        B,H,HK,D,P=64,8,2,128,16
        q=torch.randn(q_tokens,H,D,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(tokens,HK,D,device='cuda',dtype=torch.bfloat16)
        pages=k.reshape(-1,P,HK,D).contiguous()
        # Nonidentity page mapping is essential: contiguous K alone misses bugs.
        permutation=torch.randperm(pages.shape[0],device='cuda')
        storage=torch.empty_like(pages); storage[permutation]=pages
        table=permutation.int()
        logits=torch.einsum('qhd,khd->hqk',q.float(),k.repeat_interleave(H//HK,dim=1).float())/(D**.5)
        logits.masked_fill_(torch.arange(tokens,device='cuda')[None,None,:] >
            torch.arange(tokens-q_tokens,tokens,device='cuda')[None,:,None],-torch.inf)
        probs=logits.softmax(-1)
        dense_lse=logits.logsumexp(-1).reshape(H,q_tokens//B,B).permute(1,0,2)
        masses=probs.reshape(H,q_tokens//B,B,tokens//B,B).sum((2,4)).permute(1,2,0)
        for p in [.7,.87,.92]:
            mask,lse=kernel.get_top_p_metas(q,storage,table,tokens,B,p)
            torch.testing.assert_close(lse,dense_lse,atol=2e-3,rtol=2e-4)
            retained=(masses*mask).sum(1)
            assert retained.min() >= p*B-2e-3
            expected=kernel.top_p_baseline(masses,p,B)
            assert torch.equal(mask,expected), f'top-p differs: {tokens,q_tokens,p}'
            # Check every stored probability against the unrenormalized dense map.
            offsets=torch.full(mask.shape,-1,device='cuda',dtype=torch.int32)
            indices=mask.nonzero(); offsets[mask]=torch.arange(len(indices),device='cuda',dtype=torch.int32)
            values=torch.empty(len(indices),B,B,device='cuda',dtype=torch.bfloat16)
            kernel.sparse_extract_triton(q,storage,table,tokens,offsets,lse,values)
            ref=torch.stack([probs[h,qi*B:(qi+1)*B,ki*B:(ki+1)*B] for qi,ki,h in indices.tolist()])
            torch.testing.assert_close(values.float(),ref,atol=1e-3,rtol=1e-2)
            report['cases'].append({'tokens':tokens,'query_tokens':q_tokens,'top_p':p,
                'selected_blocks':len(indices),'max_lse_error':float((lse-dense_lse).abs().max()),
                'max_probability_error':float((values.float()-ref).abs().max()),'passed':True})
    report['passed']=True
    (Path(__file__).parent/'checks/compression_kernel.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
