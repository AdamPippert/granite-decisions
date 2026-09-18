"""Train/save/reload frozen heads, and collect untuned/trained paired predictions.
Uses synthetic fixtures only. Does not call Jev; pass saved Jev responses to the
separate compare command. Each run gets a new directory and retained server logs.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import time
from granite_decisions.llamacpp import NativeClient, LlamaRuntime, bind_runtime, gpu_offload_evidence, BackendError
from granite_decisions.training import train_bundle, read_dataset
from granite_decisions.contracts import questions


@contextmanager
def server(binary, model, mode, port, out):
    command = [str(binary), '--model', str(model), '--host','127.0.0.1','--port',str(port),
               '--ctx-size','4096','--batch-size','4096','--ubatch-size','4096','--parallel','1',
               '--n-gpu-layers','99','--threads','4','--threads-batch','4','--no-context-shift',
               '--no-webui','--cors-origins','localhost','--verbosity','4']
    if mode == 'embedding': command += ['--embedding','--pooling','last']
    with (out / (mode + '.log')).open('x') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    try:
        client = NativeClient(f'http://127.0.0.1:{port}',timeout=120)
        deadline = time.monotonic()+120
        while True:
            if proc.poll() is not None: raise RuntimeError(f'{mode} server exited; inspect log')
            try:
                if client.request('/health').get('status') == 'ok': break
            except BackendError: pass
            if time.monotonic()>deadline: raise RuntimeError('server health timeout')
            time.sleep(.25)
        evidence = gpu_offload_evidence((out/(mode+'.log')).read_text())
        (out/(mode+'-offload.json')).write_text(json.dumps(evidence,indent=2)+'\n')
        manifest = bind_runtime(client,str(model),mode=mode)
        runtime_path = out/(mode+'-runtime.json')
        runtime_path.write_text(json.dumps(manifest,indent=2)+'\n')
        yield LlamaRuntime(client,manifest),runtime_path
    finally:
        proc.terminate()
        try: proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server',required=True)
    p.add_argument('--gguf',required=True)
    p.add_argument('--data',required=True)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    out=Path(args.out).resolve(); out.mkdir(parents=True,exist_ok=False)
    data=Path(args.data).resolve()
    qs=questions(json.loads((data/'questions.json').read_text()))
    splits=[read_dataset(data/(s+'.jsonl'),qs) for s in ('train','calibration','test')]
    def collect(runtime_path,port,name,bundle=None):
        command=[sys.executable,str(Path(__file__).with_name('benchmark_decisions.py')),'native',
                 '--questions',str(data/'questions.json'),'--data',str(data/'test.jsonl'),
                 '--runtime',str(runtime_path),'--url',f'http://127.0.0.1:{port}','--out',str(out/(name+'.jsonl'))]
        if bundle: command += ['--bundle',str(bundle)]
        subprocess.run(command,check=True)
    with server(Path(args.server).resolve(),Path(args.gguf).resolve(),'embedding',19091,out) as (runtime,path):
        result=train_bundle(runtime,qs,*splits,out/'heads')
        (out/'training.json').write_text(json.dumps(result,indent=2)+'\n')
        collect(path,19091,'heads',out/'heads')
    with server(Path(args.server).resolve(),Path(args.gguf).resolve(),'baseline',19092,out) as (_,path):
        collect(path,19092,'untuned')
    (out/'status.json').write_text(json.dumps({'status':'passed','quality_benchmark':False,'synthetic':True})+'\n')


if __name__=='__main__': main()
