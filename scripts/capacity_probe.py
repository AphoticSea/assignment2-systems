"""Symbolic saved-tensor storage accounting; allocates no real model weights."""
import json
from pathlib import Path

import torch
from cs336_basics.model import BasicsTransformerLM
from cs336_systems.benchmark import SIZES


def estimate(size, context, dtype):
    d, f, layers, heads = SIZES[size]
    with torch.device("meta"):
        model = BasicsTransformerLM(10000, context, d, layers, heads, f, 10000.0)
        params = sum(p.numel() for p in model.parameters())
        if dtype == "bf16":
            model = model.bfloat16()
        excluded = {p.untyped_storage()._cdata for p in model.parameters()}
        excluded |= {p.untyped_storage()._cdata for p in model.buffers()}
        saved = {}
        def pack(tensor):
            storage = tensor.untyped_storage()
            if storage._cdata not in excluded:
                saved[storage._cdata] = storage
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
            logits = model(torch.randint(0, 10000, (4, context)))
        saved[logits.untyped_storage()._cdata] = logits.untyped_storage()
        activation_bytes = sum(storage.nbytes() for storage in saved.values())
    return dict(parameters=params, parameter_bytes=params*4, saved_storage_bytes=activation_bytes,
                forward_lower_bound_bytes=params*4+activation_bytes,
                note="Unique saved storages + logits. BF16 uses a fully BF16 symbolic model as a conservative activation bound; actual mixed precision retains FP32 parameters and some FP32 activations.")


if __name__ == "__main__":
    result = {f"{size}_ctx{context}_{dtype}": estimate(size, context, dtype)
              for size in list(SIZES)[1:] for context in [128,256,512,1024] for dtype in ["fp32","bf16"]}
    Path("results/steps456/capacity.json").write_text(json.dumps(result,indent=2)+"\n")
    for key, value in result.items():
        print(key, round(value["forward_lower_bound_bytes"]/1024**3, 2), "GiB")
