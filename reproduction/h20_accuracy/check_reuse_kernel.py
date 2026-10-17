"""Compare the actual BSR head-transformation kernel with dense mapped attention."""
import importlib.util
import json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('reuse_kernel_under_test',ROOT/'mapflow/kernel/bsr_varlen_page_kernel_search_k.py')
kernel=importlib.util.module_from_spec(spec);spec.loader.exec_module(kernel)


def main():
    torch.manual_seed(37)
    cases=[]
    for pair,M,N in [('Qwen3-8B_to_Qwen3-1.7B',32,16),('Qwen3-14B_to_Qwen3-8B',40,32)]:
        L,B,D,P,HK=256,64,128,16,8
        logits=torch.randn(M,L,L,device='cuda')
        logits.masked_fill_(torch.arange(L,device='cuda')[None,None,:]>torch.arange(L,device='cuda')[None,:,None],-torch.inf)
        attention=logits.softmax(-1).bfloat16()
        sparse=torch.zeros_like(attention)
        blocks=[];cols=[];crow=[0]
        for qi in range(L//B):
            for ki in range(qi+1):
                for h in range(M):
                    if ki==qi or (qi+ki+h)%3:
                        block=attention[h,qi*B:(qi+1)*B,ki*B:(ki+1)*B]
                        blocks.append(block);cols.append(ki*M+h)
                        sparse[h,qi*B:(qi+1)*B,ki*B:(ki+1)*B]=block
            crow.append(len(cols))
        data=torch.stack(blocks)
        # Deliberately shuffle both the attention block storage and V page table.
        block_indices=torch.randperm(len(data),device='cuda')
        stored=torch.empty_like(data);stored[block_indices]=data
        value=torch.randn(L,HK,D,device='cuda',dtype=torch.bfloat16)
        pages=value.reshape(-1,P,HK,D);page_order=torch.randperm(L//P,device='cuda')
        value_cache=torch.empty_like(pages);value_cache[page_order]=pages
        w=torch.load(ROOT/'examples/weights'/pair/'hotpotqa/translator_layer_3.pt',map_location='cuda',weights_only=True)['weight'].softmax(0).bfloat16()
        mixed=torch.einsum('mn,mqk->nqk',w.float(),sparse.float())
        expected=torch.einsum('nqk,knd->qnd',mixed,value.repeat_interleave(N//HK,1).float())
        ints=lambda x:torch.tensor(x,device='cuda',dtype=torch.int32)
        for max_split_k in [1,32]:
            output=torch.full((L,N,D),float('nan'),device='cuda',dtype=torch.bfloat16)
            kernel.bsr_varlen_page_triton(data_blocks=stored,block_indices=block_indices.int(),
                crow=ints(crow),col=ints(cols),cu_crow_indices=ints([0,len(crow)]),
                cu_col_indices=ints([0,len(cols)]),bsr_batch_offsets=ints([0]),
                num_packed_block_rows=L//B,packed_row_block_pid_to_seq_id=ints([0]*(L//B)),
                packed_row_block_pid_to_row_block_pid=ints(list(range(L//B))),weights=w,
                block_table=page_order.int()[None,:],value_cache=value_cache,output=output,
                density=70,max_split_k=max_split_k)
            assert torch.isfinite(output).all()
            difference=output.float()-expected
            relative_rms=float(difference.square().mean().sqrt()/expected.square().mean().sqrt())
            assert relative_rms<.025,(pair,max_split_k,relative_rms)
            torch.testing.assert_close(output.float(),expected,atol=.025,rtol=.06)
            case={'pair':pair,'max_split_k':max_split_k,'relative_rms':relative_rms,
                  'max_error':float(difference.abs().max()),'passed':True}
            cases.append(case);print(json.dumps(case),flush=True)
    (Path(__file__).parent/'checks/reuse_kernel.json').write_text(json.dumps({'passed':True,'cases':cases},indent=2)+'\n')


if __name__=='__main__':main()
