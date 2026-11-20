import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/steps78'/os.environ.get('CS336_RUN_NAME','local')/'models'
OUT.mkdir(parents=True,exist_ok=True)
ENV={**os.environ,'TORCHINDUCTOR_COMPILE_THREADS':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
     'TORCHINDUCTOR_CACHE_DIR':str(Path.home()/'.cache/cs336/inductor')}
for context in [128,512]:
    for dtype in ['fp32','bf16']:
        for mode in ['forward','train']:
            for method in ['eager','compiled']:
                name=f'{method}_small_s{context}_{dtype}_{mode}'
                print('START',name,flush=True)
                command=[sys.executable,'-m','cs336_systems.model_compile_benchmark','--method',method,'--context-length',str(context),
                         '--dtype',dtype,'--mode',mode,'--output',str(OUT/(name+'.json'))]
                started=time.time()
                with (OUT/(name+'.log')).open('w') as log:
                    proc=subprocess.run(command,cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,timeout=900)
                print('END',name,proc.returncode,round(time.time()-started,1),'s',flush=True)
