"""NVTX annotation of the unchanged mathematical operations in staff attention."""
import math

import torch
from einops import einsum
import cs336_basics.model as staff


def annotated_attention(Q, K, V, mask=None):
    with torch.cuda.nvtx.range("attention_qk"):
        scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(K.shape[-1])
    with torch.cuda.nvtx.range("attention_mask_softmax"):
        if mask is not None:
            scores = torch.where(mask, scores, float("-inf"))
        weights = staff.softmax(scores, dim=-1)
    with torch.cuda.nvtx.range("attention_pv"):
        return einsum(weights, V, "... query key, ... key d_v -> ... query d_v")


def annotate_attention():
    staff.scaled_dot_product_attention = annotated_attention
