"""Milestones 7/8: isolated attention timing with compilation excluded."""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import traceback

import torch
import triton
import torch.nn.functional as F

from cs336_systems.flash_attention import FlashAttentionTriton


def eager_attention(q,k,v,is_causal=False):
    scores=q@k.transpose(-1,-2)/math.sqrt(q.shape[-1])
    if is_causal:
        mask=torch.arange(q.shape[-2],device=q.device)[:,None]>=torch.arange(k.shape[-2],device=q.device)[None,:]
        scores=scores.masked_fill(~mask,-torch.inf)
    return torch.softmax(scores,dim=-1)@v


def sdpa_attention(q,k,v,is_causal=False):
    return F.scaled_dot_product_attention(q.unsqueeze(1),k.unsqueeze(1),v.unsqueeze(1),is_causal=is_causal).squeeze(1)


def check_rows(output,q,k,v,causal):
    nq=q.shape[-2]
    indices=torch.linspace(0,nq-1,min(8,nq),device=q.device).long()
    scores=q.detach()[:,indices].float()@k.detach().float().transpose(-1,-2)/q.shape[-1]**0.5
    if causal:
        scores=scores.masked_fill(indices[:,None]<torch.arange(k.shape[-2],device=q.device)[None,:],-torch.inf)
    reference=torch.softmax(scores,-1)@v.detach().float()
    tol=0.02 if q.dtype==torch.bfloat16 else 3e-4
    actual=output.detach()[:,indices].float()
    torch.testing.assert_close(actual,reference,atol=tol,rtol=tol)
    return {'sampled_rows':len(indices),'max_absolute_error':(actual-reference).abs().max().item(),
            'relative_l2':((actual-reference).norm()/reference.norm()).item()}


def run_case(fn,method,n,d,dtype,batch,causal,backward,warmup,iterations,target):
    torch.manual_seed(336)
    result=dict(status='running',method=method,seq_len=n,head_dim=d,dtype=str(dtype),batch_size=batch,
                is_causal=causal,backward_measured=backward,warmup=warmup,iterations=iterations,
                torch=torch.__version__,triton=triton.__version__,gpu=torch.cuda.get_device_name(),tf32=False,
                allocator_memory_fraction=.90,
                benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                flash_sha256=hashlib.sha256(Path(__file__).with_name('flash_attention.py').read_bytes()).hexdigest())
    try:
        q,k,v=[torch.randn(batch,n,d,device='cuda',dtype=dtype,requires_grad=True) for _ in range(3)]
        do=torch.randn_like(q)
        def step():
            for x in (q,k,v):x.grad=None
            torch.cuda.synchronize();start=time.perf_counter()
            output=fn(q,k,v,causal)
            torch.cuda.synchronize();middle=time.perf_counter()
            before_backward=torch.cuda.memory_allocated()
            backward_ms=None
            if backward:
                bstart=time.perf_counter();output.backward(do);torch.cuda.synchronize()
                backward_ms=(time.perf_counter()-bstart)*1000
            return output,{'forward':(middle-start)*1000,'backward':backward_ms,
                           'total':(middle-start)*1000+(backward_ms or 0),
                           'allocated_before_backward_mib':before_backward/1024**2}
        cold=time.perf_counter()
        output,_=step()
        result['first_call_including_compile_seconds']=time.perf_counter()-cold
        result['numerics']=check_rows(output,q,k,v,causal)
        del output
        for _ in range(warmup):
            output,_=step();del output
        for x in (q,k,v):x.grad=None
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        samples=[]
        for _ in range(iterations):
            output,sample=step();samples.append(sample);del output
        result['samples_ms']=samples
        result['summary_ms']={key:{'mean':statistics.mean(s[key] for s in samples),
                                   'stdev':statistics.stdev(s[key] for s in samples) if iterations>1 else 0}
                              for key in ['forward','total']+(['backward'] if backward else [])}
        result['allocated_before_backward_mib']=statistics.mean(s['allocated_before_backward_mib'] for s in samples)
        result['peak_allocated_mib']=torch.cuda.max_memory_allocated()/1024**2
        result['status']='passed'
        if method=='sdpa_auto':
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
                output=fn(q,k,v,causal);torch.cuda.synchronize()
            result['sdpa_operator_names']=[e.key for e in profiler.key_averages() if 'scaled_dot_product' in e.key]
            del output
    except Exception as error:
        result.update(status='oom' if isinstance(error,torch.OutOfMemoryError) or 'out of memory' in str(error).lower() else 'error',
                      error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc()[-5000:])
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(target.name,result['status'],result.get('summary_ms',{}).get('forward',{}).get('mean'),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--suite',choices=['attention','flash'],required=True)
    parser.add_argument('--head-dim',type=int,required=True)
    parser.add_argument('--dtype',choices=['fp32','bf16'],required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=100)
    args=parser.parse_args()
    torch.cuda.set_per_process_memory_fraction(.90)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    dtype=torch.float32 if args.dtype=='fp32' else torch.bfloat16
    batch,causal,backward=(8,False,True) if args.suite=='attention' else (1,True,False)
    lengths=[256,1024,4096,8192,16384] if args.suite=='attention' else [128,512,2048,4096,8192,16384]
    methods=['eager','compiled'] if args.suite=='attention' else ['eager','compiled','sdpa_auto','triton']
    compiled=torch.compile(eager_attention,fullgraph=True,dynamic=False)
    funcs={'eager':eager_attention,'compiled':compiled,'sdpa_auto':sdpa_attention,'triton':FlashAttentionTriton.apply}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for n in lengths:
        for method in methods:
            target=args.output_dir/f'{method}_b{batch}_s{n}_d{args.head_dim}_{args.dtype}.json'
            if target.exists():
                old=json.loads(target.read_text())
                valid=old.get('benchmark_sha256')==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                valid=valid and old.get('iterations')==args.iterations and old.get('gpu')==torch.cuda.get_device_name()
                if method=='triton':valid=valid and old.get('flash_sha256')==hashlib.sha256(Path(__file__).with_name('flash_attention.py').read_bytes()).hexdigest()
                if valid and old.get('status') in ['passed','oom']:
                    print('cached',target.name,flush=True);continue
            run_case(funcs[method],method,n,args.head_dim,dtype,batch,causal,backward,5,args.iterations,target)
            gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
