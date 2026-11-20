"""2025 A2 torch_compile(b): wrap the staff Transformer, keeping AdamW unchanged."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import statistics
import time
import traceback

import torch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--method',choices=['eager','compiled'],required=True)
    parser.add_argument('--context-length',type=int,required=True)
    parser.add_argument('--dtype',choices=['fp32','bf16'],required=True)
    parser.add_argument('--mode',choices=['forward','train'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fingerprint=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.output.exists():
        old=json.loads(args.output.read_text())
        if old.get('source_sha256')==fingerprint and old.get('status')=='passed':
            print('cached',args.output.name);return
    torch.cuda.set_per_process_memory_fraction(.9)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(336)
    result=dict(status='running',method=args.method,size='small',context_length=args.context_length,
                dtype=args.dtype,mode=args.mode,batch_size=4,source_sha256=fingerprint,
                torch=torch.__version__,gpu=torch.cuda.get_device_name(),warmup=5,iterations=10,
                allocator_memory_fraction=.9,tf32=False,fullgraph=False,
                timing='synchronized phase wall times; zero_grad excluded; compile excluded')
    try:
        with torch.device('cuda'):
            model=BasicsTransformerLM(10000,args.context_length,768,12,12,3072,10000.0).train()
            inputs=torch.randint(0,10000,(4,args.context_length))
            targets=torch.randint(0,10000,inputs.shape)
        result['parameters']=sum(p.numel() for p in model.parameters())
        def autocast():return torch.autocast('cuda',dtype=torch.bfloat16) if args.dtype=='bf16' else nullcontext()
        with autocast():reference=model(inputs).detach()
        function=torch.compile(model) if args.method=='compiled' else model
        optimizer=AdamW(model.parameters(),lr=1e-4) if args.mode=='train' else None
        def step():
            model.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
            with autocast():output=function(inputs)
            torch.cuda.synchronize();fend=time.perf_counter()
            row={'forward':(fend-start)*1000}
            if optimizer:
                loss=cross_entropy(output.float(),targets);loss.backward()
                torch.cuda.synchronize();bend=time.perf_counter()
                optimizer.step();torch.cuda.synchronize();oend=time.perf_counter()
                row.update(loss_backward=(bend-fend)*1000,optimizer=(oend-bend)*1000)
                if not torch.isfinite(loss).item():raise RuntimeError('non-finite loss')
            row['total']=sum(row.values())
            return output,row
        first=time.perf_counter();output,_=step()
        result['first_call_including_compile_seconds']=time.perf_counter()-first
        tolerance=.02 if args.dtype=='bf16' else 3e-4
        torch.testing.assert_close(output,reference,atol=tolerance,rtol=tolerance)
        result['output_max_absolute_error']=(output-reference).abs().max().item()
        del reference,output
        for _ in range(5):output,_=step();del output
        model.zero_grad(set_to_none=True);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        samples=[]
        for _ in range(10):output,row=step();samples.append(row);del output
        result['samples_ms']=samples
        result['summary_ms']={k:{'mean':statistics.mean(r[k] for r in samples),'stdev':statistics.stdev(r[k] for r in samples)} for k in samples[0]}
        result['peak_allocated_mib']=torch.cuda.max_memory_allocated()/1024**2
        result['dynamo_counters']={k:dict(v) for k,v in torch._dynamo.utils.counters.items() if v}
        result['status']='passed'
    except Exception as error:
        result.update(status='oom' if isinstance(error,torch.OutOfMemoryError) or 'out of memory' in str(error).lower() else 'error',error=str(error),traceback=traceback.format_exc()[-6000:])
    args.output.write_text(json.dumps(result,indent=2,default=str)+'\n')
    print(args.output.name,result['status'],result.get('summary_ms',{}).get('total',{}).get('mean'),flush=True)


if __name__=='__main__':main()
