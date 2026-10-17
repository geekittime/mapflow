"""Launch one audited runtime component; source env.sh first.

Uses installed vLLM+overlay, never the vendored source as an import root.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
PAIRS={'8_17':('Qwen3-8B','Qwen3-1.7B',36,28,32,16),
       '14_8':('Qwen3-14B','Qwen3-8B',40,36,40,32)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('role',choices=['server','sender','receiver','native'])
    p.add_argument('--pair',choices=PAIRS,required=True)
    p.add_argument('--task',choices=['hotpotqa','conversation','multinews','rulerqa'],required=True)
    p.add_argument('--devices',required=True)
    p.add_argument('--base-port',type=int,default=7000)
    p.add_argument('--run-dir',type=Path,required=True)
    a=p.parse_args();a.run_dir=a.run_dir.resolve();a.run_dir.mkdir(parents=True,exist_ok=True)
    assert a.run_dir.is_relative_to(Path('/home/tiger'))
    s,t,sl,tl,sh,th=PAIRS[a.pair]
    top_p=(.92 if a.pair=='8_17' else .87) if a.task=='rulerqa' else .7
    ignored=sl-tl+3
    env={'CUDA_VISIBLE_DEVICES':a.devices,'PYTHONPATH':str(ROOT), 'PYTHONUNBUFFERED':'1',
         'VLLM_ENABLE_V1_MULTIPROCESSING':'0','OMP_NUM_THREADS':'4',
         'VLLM_ATTENTION_BACKEND':'FLASH_ATTN' if a.role=='native' else 'BSR_ATTN'}
    # Prevent inherited MapFlow settings from contaminating native baselines.
    for key in list(os.environ):
        if key.startswith(('VLLM_NUM_','VLLM_ATTNMAP_','VLLM_BSR_')) or key in ['MAPFLOW_AUDIT_FILE','SKIP_DENSITY_CHECK']:
            del os.environ[key]
    if a.role=='server':
        env['BLOCK_CACHE_POLICY']='FIFO'
        cmd=[sys.executable,'-m','mapflow.server.server','--num_sender_layers',str(sl),
             '--num_ignored_layers',str(ignored),'--num_sender_heads',str(sh),'--tp_size','1',
             '--block_size','64','--max_num_reqs','128','--dtype','bfloat16','--port',str(a.base_port)]
    else:
        sender=a.role=='sender'
        if a.role!='native':
            env.update({k:str(v) for k,v in dict(VLLM_NUM_KV_HEADS=8,VLLM_HEAD_DIM=128,
                VLLM_MAX_SEQ_LEN=40960,VLLM_BLOCK_SIZE=64,VLLM_NUM_LAYERS=sl if sender else tl,
                VLLM_NUM_SEND_LAYERS=sl,VLLM_NUM_HEADS=sh if sender else th,
                VLLM_NUM_SEND_HEADS=sh,VLLM_NUM_IGNORED_LAYERS=ignored if sender else 3,
                VLLM_BSR_TOP_P=top_p,VLLM_SERVER_MAIN_PORT=a.base_port).items()})
        if a.role=='receiver':
            env.update(VLLM_ATTNMAP_WEIGHTS_DIR=str(ROOT/'examples/weights'/f'{s}_to_{t}'/a.task),
                       VLLM_ATTNMAP_FORCE_WAIT_SYNC='1',SKIP_DENSITY_CHECK='1',
                       MAPFLOW_AUDIT_FILE=str(a.run_dir/'reuse_events.jsonl'))
        port=a.base_port+{'sender':1,'receiver':2,'native':3}[a.role]
        cmd=[sys.executable,'-m','vllm.entrypoints.openai.api_server','--model',str(ROOT/'examples/models'/(s if sender else t)),
             '--served-model-name',s if sender else t,'--host','127.0.0.1','--port',str(port),
             '--enforce-eager','--no-async-scheduling','--disable-log-stats','--tensor-parallel-size','1',
             '--max-model-len',str(20480 if sender else 40960),
             '--gpu-memory-utilization',str(.9 if sender else .5),
             '--max-num-batched-tokens',str(4096 if sender else 4116)]
        if not sender:cmd+=['--no-enable-prefix-caching']
    metadata={'pid':os.getpid(),'role':a.role,'pair':a.pair,'task':a.task,'command':cmd,'environment':env,
              'attention_top_p':top_p,'mps':False,'notes':'Accuracy only; no MPS thread limits; default author cache capacities.',
              'launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (a.run_dir/f'{a.role}.launch.json').write_text(json.dumps(metadata,indent=2)+'\n')
    os.environ.update(env)
    os.chdir(ROOT)
    os.execv(sys.executable,cmd)


if __name__=='__main__':main()
