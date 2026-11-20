"""Validate CUDA, Triton JIT and the A1 staff model before A2 benchmarking."""
import argparse
import json
import platform
from pathlib import Path

import torch
import triton
import triton.language as tl

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy


@triton.jit
def add_kernel(x, y, out, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(out + offsets, tl.load(x + offsets, mask, other=0) + tl.load(y + offsets, mask, other=0), mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/environment.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this check must run in the WSL CUDA environment.")
    torch.manual_seed(336)
    x = torch.randn(1003, device="cuda")
    y = torch.randn_like(x)
    out = torch.empty_like(x)
    add_kernel[(triton.cdiv(x.numel(), 256),)](x, y, out, x.numel(), BLOCK=256)
    torch.testing.assert_close(out, x + y)

    model = BasicsTransformerLM(1000, 64, 128, 2, 4, 512, 10000.0).cuda()
    # Constructor takes vocab_size, context_length, d_model, num_layers,
    # num_heads, d_ff, rope_theta in that order.
    inputs = torch.randint(0, 1000, (2, 64), device="cuda")
    targets = torch.randint(0, 1000, (2, 64), device="cuda")
    loss = cross_entropy(model(inputs), targets)
    loss.backward()
    assert torch.isfinite(loss).item(), "Non-finite loss"
    assert all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters())
    torch.cuda.synchronize()
    gpu = torch.cuda.get_device_properties(0)
    result = {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "triton": triton.__version__, "gpu": gpu.name,
        "gpu_memory_mib": gpu.total_memory / 1024**2,
        "compute_capability": list(torch.cuda.get_device_capability()),
        "triton_vector_add": "passed", "staff_model_forward_backward": "passed",
        "smoke_loss": loss.item(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
