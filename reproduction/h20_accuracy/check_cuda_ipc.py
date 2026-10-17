"""Exercise the author's IPC wrapper across processes and visible GPU remapping."""
import importlib.util
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def load_wrapper():
    spec=importlib.util.spec_from_file_location('ipc_under_test',ROOT/'mapflow/core/ipc_utils.py')
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module.CudaIPCWrapper


def receive():
    import torch
    load_wrapper()
    # Deliberately use a cold CUDA context to cover freshly spawned workers.
    assert not torch.cuda.is_initialized()
    wrapper=pickle.loads(sys.stdin.buffer.read())
    print('IPC receiver: reconstruct',file=sys.stderr,flush=True)
    t=wrapper.to_tensor(device_idx=0)
    print('IPC receiver: compare',file=sys.stderr,flush=True)
    expected=torch.arange(16,80,device='cuda',dtype=torch.bfloat16)
    torch.testing.assert_close(t,expected,atol=0,rtol=0)
    t.add_(1)
    torch.cuda.synchronize()
    print('IPC receiver: synchronized',file=sys.stderr,flush=True)
    print(json.dumps({'read_passed':True,'write_passed':True,'visible':os.environ['CUDA_VISIBLE_DEVICES']}))


def send(destination):
    import torch
    Wrapper=load_wrapper()
    base=torch.arange(128,device='cuda',dtype=torch.bfloat16)
    t=base[16:80]
    wrapper=Wrapper(t)
    torch.cuda.synchronize()
    env={**os.environ,'CUDA_VISIBLE_DEVICES':destination}
    child=subprocess.run([sys.executable,__file__,'receive'],input=pickle.dumps(wrapper),
                         capture_output=True,env=env,timeout=45)
    if child.returncode:
        raise RuntimeError(f'receiver exit={child.returncode}, stderr={child.stderr.decode()}')
    torch.cuda.synchronize()
    torch.testing.assert_close(t,torch.arange(17,81,device='cuda',dtype=torch.bfloat16),atol=0,rtol=0)
    print(json.dumps({'source':os.environ['CUDA_VISIBLE_DEVICES'],'destination':destination,
                      'passed':True,'child':json.loads(child.stdout),'torch':torch.__version__}))


def main():
    cases=[]
    for source,dest in [(0,1),(1,2),(2,3),(3,0)]:
        child=subprocess.run([sys.executable,__file__,'send',str(dest)],capture_output=True,
                             env={**os.environ,'CUDA_VISIBLE_DEVICES':str(source)},timeout=55)
        if child.returncode:
            raise RuntimeError(f'sender exit={child.returncode}, stderr={child.stderr.decode()}')
        cases.append(json.loads(child.stdout))
        print(json.dumps(cases[-1]),flush=True)
    (Path(__file__).parent/'checks/cuda_ipc.json').write_text(json.dumps({'passed':True,'cases':cases},indent=2)+'\n')


if __name__=='__main__':
    if len(sys.argv)==1: main()
    elif sys.argv[1]=='receive': receive()
    else: send(sys.argv[2])
