"""Numerical and layout checks beyond the original forward smoke tests."""
import pytest
import torch
from cs336_systems.flash_attention import FlashAttentionPytorch,FlashAttentionTriton


@pytest.mark.parametrize('nq,nk,d',[(16,16,16),(47,63,32),(129,193,64),(257,257,128),(65,97,96)])
@pytest.mark.parametrize('causal',[False,True])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_forward_shapes_layouts(nq,nk,d,causal,dtype):
    torch.manual_seed(336)
    q=torch.randn(2,nq,d*2,device='cuda',dtype=dtype)[...,::2].requires_grad_()
    k=torch.randn(2,nk,d*2,device='cuda',dtype=dtype)[...,::2].requires_grad_()
    v=torch.randn_like(k).requires_grad_()
    scores=q.float()@k.float().transpose(-1,-2)/d**0.5
    if causal:
        scores=scores.masked_fill(torch.arange(nq,device='cuda')[:,None]<torch.arange(nk,device='cuda')[None,:],-torch.inf)
    expected=torch.softmax(scores,-1)@v.float()
    expected_lse=torch.logsumexp(scores,-1)
    for cls in [FlashAttentionPytorch,FlashAttentionTriton]:
        actual=cls.apply(q,k,v,causal)
        lse=[x for x in actual.grad_fn.saved_tensors if x.shape==q.shape[:-1]][0]
        tol=0.015 if dtype==torch.bfloat16 else 2e-4
        torch.testing.assert_close(actual.float(),expected,atol=tol,rtol=tol)
        torch.testing.assert_close(lse,expected_lse,atol=2e-4,rtol=2e-4)
        assert torch.isfinite(actual).all()


def test_backward_is_explicitly_pending():
    inputs=[torch.randn(1,32,16,device='cuda',requires_grad=True) for _ in range(3)]
    for cls in [FlashAttentionPytorch,FlashAttentionTriton]:
        with pytest.raises(NotImplementedError,match='milestone 9'):
            cls.apply(*inputs).sum().backward()
