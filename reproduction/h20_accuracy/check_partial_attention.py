"""GPU regression: miss-only FA3 batch must not reuse full-batch scheduling."""
import json
from pathlib import Path
import torch
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func,get_scheduler_metadata


def main():
    torch.manual_seed(31)
    L,H,HK,D,P=256,16,8,128,16
    q=torch.randn(L,H,D,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(L,HK,D,device='cuda',dtype=torch.bfloat16)
    v=torch.randn_like(k)
    cache_k=k.reshape(-1,P,HK,D).contiguous();cache_v=v.reshape_as(cache_k).contiguous()
    block_table=torch.arange(L//P,device='cuda',dtype=torch.int32)[None,:]
    logits=torch.einsum('qhd,khd->hqk',q.float(),k.repeat_interleave(H//HK,1).float())/(D**.5)
    logits.masked_fill_(torch.arange(L,device='cuda')[None,None,:]>torch.arange(L,device='cuda')[None,:,None],-torch.inf)
    reference=torch.einsum('hqk,khd->qhd',logits.softmax(-1),v.repeat_interleave(H//HK,1).float())
    checks=[]
    for spans in [[(192,256)],[(0,64),(192,256)],[(64,128),(192,256)]]:
        subset=torch.cat([q[a:b] for a,b in spans]);out=torch.empty_like(subset)
        cu=torch.tensor([0]+list(__import__('itertools').accumulate(b-a for a,b in spans)),device='cuda',dtype=torch.int32)
        lens=torch.tensor([b for a,b in spans],device='cuda',dtype=torch.int32)
        flash_attn_varlen_func(q=subset,k=cache_k,v=cache_v,out=out,cu_seqlens_q=cu,
            max_seqlen_q=max(b-a for a,b in spans),seqused_k=lens,max_seqlen_k=L,
            block_table=block_table.expand(len(spans),-1).contiguous(),causal=True,
            scheduler_metadata=None,fa_version=3)
        expected=torch.cat([reference[a:b] for a,b in spans])
        torch.testing.assert_close(out.float(),expected,atol=.015,rtol=.02)
        checks.append({'spans':spans,'max_error':float((out.float()-expected).abs().max()),'passed':True})
    result={'passed':True,'torch':torch.__version__,'fa_version':3,'cases':checks}
    (Path(__file__).parent/'checks/partial_attention.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
