"""Run milestones 7/8 sequentially on one GPU, preserving worker logs."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/steps78'/os.environ.get('CS336_RUN_NAME','local')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('suite',choices=['attention','flash','both'])
    args=parser.parse_args()
    env={**os.environ,'TORCHINDUCTOR_COMPILE_THREADS':'1','OMP_NUM_THREADS':'4',
         'TORCHINDUCTOR_CACHE_DIR':str(Path.home()/'.cache/cs336/inductor'),
         'WANDB_MODE':'disabled'}
    suites=['attention','flash'] if args.suite=='both' else [args.suite]
    for suite in suites:
        folder=OUT/suite;folder.mkdir(parents=True,exist_ok=True)
        for dtype in ['fp32','bf16']:
            for d in [16,32,64,128]:
                command=[sys.executable,'-u','-m','cs336_systems.attention_benchmark','--suite',suite,
                         '--head-dim',str(d),'--dtype',dtype,'--output-dir',str(folder)]
                log=folder/f'worker_d{d}_{dtype}.log'
                print('START',suite,dtype,d,flush=True)
                started=time.time()
                with log.open('w') as f:
                    proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                    try:code=proc.wait(timeout=900)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid,signal.SIGKILL);proc.wait();code='timeout'
                states={}
                for p in folder.glob(f'*_d{d}_{dtype}.json'):
                    state=json.loads(p.read_text())['status'];states[state]=states.get(state,0)+1
                print('END',suite,dtype,d,'exit',code,'seconds',round(time.time()-started,1),states,flush=True)
                if code!=0:raise RuntimeError(f'Worker failed; see {log}')


if __name__=='__main__':main()
