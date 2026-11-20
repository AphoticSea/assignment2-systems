"""2025 A2 1.3.2: tiled online-softmax forward, in PyTorch and Triton.

Backward is deliberately left for milestone 9. FP32 Triton dot uses IEEE
precision, matching the strict FP32 baseline rather than silently enabling TF32.
"""
import math

import torch
import triton
import triton.language as tl


def _validate(q, k, v):
    if q.ndim < 2 or q.shape[:-2] != k.shape[:-2] or k.shape != v.shape:
        raise ValueError("Expected matching leading dimensions and equal K/V shapes")
    if q.shape[-1] != k.shape[-1] or min(q.shape[-2], k.shape[-2], q.shape[-1]) < 1:
        raise ValueError("Attention dimensions must match and be positive")
    if q.dtype != k.dtype or q.dtype != v.dtype or q.device != k.device or q.device != v.device:
        raise ValueError("Q/K/V must have the same dtype and device")


class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        _validate(q, k, v)
        nq, d = q.shape[-2:]
        nk = k.shape[-2]
        q3, k3, v3 = [x.reshape(-1, x.shape[-2], d) for x in (q, k, v)]
        batch = q3.shape[0]
        output = torch.empty_like(q3)
        logsumexp = torch.empty((batch, nq), dtype=torch.float32, device=q.device)
        bq, bk = 64, 64
        for i in range(0, nq, bq):
            qi = q3[:, i:i+bq].float()
            rows = qi.shape[-2]
            m = torch.full((batch, rows), -torch.inf, device=q.device)
            l = torch.zeros_like(m)
            o = torch.zeros((batch, rows, d), dtype=torch.float32, device=q.device)
            last = min(nk, i + rows) if is_causal else nk
            for j in range(0, last, bk):
                scores = qi @ k3[:, j:j+bk].float().transpose(-1, -2) / math.sqrt(d)
                if is_causal:
                    mask = torch.arange(i, i+rows, device=q.device)[:, None] >= torch.arange(j, min(j+bk,nk), device=q.device)[None, :]
                    scores = scores.masked_fill(~mask, -torch.inf)
                new_m = torch.maximum(m, scores.max(dim=-1).values)
                alpha = torch.exp(m - new_m)
                p = torch.exp(scores - new_m.unsqueeze(-1))
                l = alpha * l + p.sum(dim=-1)
                o = alpha.unsqueeze(-1) * o + p.to(v.dtype).float() @ v3[:, j:j+bk].float()
                m = new_m
            output[:, i:i+rows] = (o/l.unsqueeze(-1)).to(q.dtype)
            logsumexp[:, i:i+rows] = m + torch.log(l)
        output = output.reshape(q.shape)
        logsumexp = logsumexp.reshape(q.shape[:-1])
        ctx.save_for_backward(q,k,v,output,logsumexp)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("FlashAttention backward belongs to milestone 9")


@triton.jit
def flash_forward_kernel(Q,K,V,O,L,
                         sqb,sqq,sqd,skb,skn,skd,svb,svn,svd,sob,soq,sod,slb,slq,
                         NQ:tl.constexpr,NK:tl.constexpr,D:tl.constexpr,
                         BM:tl.constexpr,BN:tl.constexpr,BD:tl.constexpr,
                         IS_CAUSAL:tl.constexpr,SCALE:tl.constexpr):
    block=tl.program_id(0)
    batch=tl.program_id(1)
    qp=tl.make_block_ptr(Q+batch*sqb,(NQ,D),(sqq,sqd),(block*BM,0),(BM,BD),(1,0))
    kp=tl.make_block_ptr(K+batch*skb,(D,NK),(skd,skn),(0,0),(BD,BN),(0,1))
    vp=tl.make_block_ptr(V+batch*svb,(NK,D),(svn,svd),(0,0),(BN,BD),(1,0))
    q=tl.load(qp,boundary_check=(0,1),padding_option="zero")
    qrows=block*BM+tl.arange(0,BM)
    keys=tl.arange(0,BN)
    m=tl.full((BM,),-float('inf'),tl.float32)
    l=tl.zeros((BM,),tl.float32)
    acc=tl.zeros((BM,BD),tl.float32)
    end=NK
    if IS_CAUSAL:
        end=tl.minimum(NK,(block+1)*BM)
    for start in range(0,end,BN):
        k=tl.load(kp,boundary_check=(0,1),padding_option="zero")
        v=tl.load(vp,boundary_check=(0,1),padding_option="zero")
        scores=tl.dot(q,k,input_precision="ieee")*SCALE
        valid=(start+keys[None,:])<NK
        if IS_CAUSAL:
            valid=valid & (qrows[:,None]>=start+keys[None,:])
        scores=tl.where(valid,scores,-float('inf'))
        new_m=tl.maximum(m,tl.max(scores,1))
        p=tl.exp(scores-new_m[:,None])
        alpha=tl.exp(m-new_m)
        l=l*alpha+tl.sum(p,1)
        acc=acc*alpha[:,None]
        acc=tl.dot(p.to(v.dtype),v,acc=acc,input_precision="ieee")
        m=new_m
        kp=tl.advance(kp,(0,BN))
        vp=tl.advance(vp,(BN,0))
    normalized=acc/l[:,None]
    op=tl.make_block_ptr(O+batch*sob,(NQ,D),(soq,sod),(block*BM,0),(BM,BD),(1,0))
    tl.store(op,normalized.to(op.type.element_ty),boundary_check=(0,1))
    tl.store(L+batch*slb+qrows*slq,m+tl.log(l),mask=qrows<NQ)


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,is_causal=False):
        _validate(q,k,v)
        if q.device.type!='cuda':
            raise ValueError('Triton FlashAttention requires CUDA')
        nq,d=q.shape[-2:]
        nk=k.shape[-2]
        q3,k3,v3=[x.reshape(-1,x.shape[-2],d) for x in (q,k,v)]
        output=torch.empty_like(q3)
        l=torch.empty((q3.shape[0],nq),device=q.device,dtype=torch.float32)
        # SM120's per-block shared-memory limit is smaller than H100's.
        # Strict FP32 with padded head dimension 128 needs a smaller tile.
        bm,bn=(32,32) if q.dtype==torch.float32 and d>64 else (64,64)
        flash_forward_kernel[(triton.cdiv(nq,bm),q3.shape[0])](
            q3,k3,v3,output,l,
            *q3.stride(),*k3.stride(),*v3.stride(),*output.stride(),*l.stride(),
            nq,nk,d,bm,bn,triton.next_power_of_2(d),bool(is_causal),1/math.sqrt(d),
            num_warps=4,num_stages=1 if q.dtype==torch.float32 and d>64 else 2)
        output=output.reshape(q.shape)
        l=l.reshape(q.shape[:-1])
        ctx.save_for_backward(l,q,k,v,output)
        ctx.is_causal=bool(is_causal)
        return output

    @staticmethod
    def backward(ctx,*grad_outputs):
        raise NotImplementedError('FlashAttention backward is milestone 9; this implementation provides forward only.')
