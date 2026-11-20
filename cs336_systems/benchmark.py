"""2025 A2 section 1.1.3: synchronized wall-clock benchmark of the staff model."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
from time import perf_counter

import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW


# (d_model, d_ff, num_layers, num_heads). tiny is a local smoke preset;
# small through 2.7B reproduce Table 1 in the 2025 handout.
SIZES = {
    "tiny": (128, 512, 2, 4),
    "small": (768, 3072, 12, 12),
    "medium": (1024, 4096, 24, 16),
    "large": (1280, 5120, 36, 20),
    "xl": (1600, 6400, 48, 25),
    "2.7B": (2560, 10240, 32, 32),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=SIZES, default="small")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=10000)
    for name in ("d-model", "d-ff", "num-layers", "num-heads"):
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--mode", choices=["forward", "forward-backward", "train"], default="forward-backward")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=336)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-capture", action="store_true")
    parser.add_argument("--annotate-attention", action="store_true")
    parser.add_argument("--memory-history", type=Path)
    parser.add_argument("--memory-fraction", type=float, default=0.90)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        parser.error("warmup must be nonnegative and iterations must be positive")
    d_model, d_ff, layers, heads = SIZES[args.size]
    config = dict(
        vocab_size=args.vocab_size, context_length=args.context_length,
        d_model=args.d_model if args.d_model is not None else d_model,
        d_ff=args.d_ff if args.d_ff is not None else d_ff,
        num_layers=args.num_layers if args.num_layers is not None else layers,
        num_heads=args.num_heads if args.num_heads is not None else heads,
        rope_theta=10000.0,
    )
    if min(config[k] for k in ("vocab_size", "context_length", "d_model", "d_ff", "num_layers", "num_heads")) < 1:
        parser.error("model dimensions must be positive")
    if args.batch_size < 1 or config["d_model"] % config["num_heads"]:
        parser.error("batch size must be positive and d_model must be divisible by num_heads")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for this benchmark")
    if not 0 < args.memory_fraction <= 1:
        parser.error("memory-fraction must be in (0,1]")
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    torch.manual_seed(args.seed)
    # Keep FP32 matmul precision explicit so later comparisons use the same baseline.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = {
        "status": "running", "config": config,
        "mode": args.mode, "dtype": args.dtype, "batch_size": args.batch_size,
        "warmup": args.warmup, "iterations": args.iterations, "seed": args.seed,
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "tf32": False,
        "grad_enabled_in_forward": True, "model_training": True,
        "timing": "perf_counter plus CUDA synchronization; zero_grad excluded",
        "is_full_assignment_sweep": False,
        "protocol_version": 3,
        "allocator_memory_fraction": args.memory_fraction,
        "initialization_device": "cuda",
        "loss_dtype": "fp32",
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "staff_model_sha256": hashlib.sha256(Path(__file__).resolve().parents[1].joinpath(
            "cs336-basics/cs336_basics/model.py").read_bytes()).hexdigest(),
    }
    try:
        result["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        result["git_head"] = None
    # Meta construction avoids allocating gigabytes of CPU RAM merely to count
    # parameters. Mixed precision retains FP32 master weights and gradients.
    with torch.device("meta"):
        meta_model = BasicsTransformerLM(**config)
    parameter_count = sum(p.numel() for p in meta_model.parameters())
    del meta_model
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    resident_copies = {"forward": 1, "forward-backward": 2, "train": 4}[args.mode]
    minimum_bytes = parameter_count * 4 * resident_copies
    result.update(parameters=parameter_count, minimum_resident_bytes=minimum_bytes,
                  free_device_bytes=free_bytes, total_device_bytes=total_bytes)
    if minimum_bytes > min(free_bytes, int(total_bytes := result["total_device_bytes"] * args.memory_fraction)):
        result.update(status="capacity_excluded", reason="FP32 weights/gradients/Adam states lower bound exceeds free device memory; activations excluded")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        raise SystemExit(3)
    if args.annotate_attention:
        from cs336_systems.profiling import annotate_attention
        annotate_attention()
    history_started = False
    capture_started = False
    try:
        with torch.device("cuda"):
            model = BasicsTransformerLM(**config).train()
        result["parameters"] = sum(p.numel() for p in model.parameters())
        inputs = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device="cuda")
        targets = torch.randint(0, args.vocab_size, inputs.shape, device="cuda")
        optimizer = AdamW(model.parameters(), lr=1e-4) if args.mode == "train" else None

        def step():
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            start = perf_counter()
            with torch.cuda.nvtx.range("forward"):
                precision = torch.autocast("cuda", dtype=torch.bfloat16) if args.dtype == "bf16" else nullcontext()
                with precision:
                    logits = model(inputs)
            torch.cuda.synchronize()
            end_forward = perf_counter()
            timings = {"forward": (end_forward - start) * 1000}
            loss = None
            if args.mode != "forward":
                with torch.cuda.nvtx.range("loss_backward"):
                    loss = cross_entropy(logits.float(), targets)
                    if args.memory_history:
                        memory_stages.append({"stage": "before_backward", "allocated_mib": torch.cuda.memory_allocated() / 1024**2})
                    loss.backward()
                torch.cuda.synchronize()
                end_backward = perf_counter()
                timings["loss_backward"] = (end_backward - end_forward) * 1000
                if args.memory_history:
                    memory_stages.append({"stage": "after_backward", "allocated_mib": torch.cuda.memory_allocated() / 1024**2})
                if optimizer is not None:
                    with torch.cuda.nvtx.range("optimizer"):
                        optimizer.step()
                    torch.cuda.synchronize()
                    timings["optimizer"] = (perf_counter() - end_backward) * 1000
                    if args.memory_history:
                        memory_stages.append({"stage": "after_optimizer", "allocated_mib": torch.cuda.memory_allocated() / 1024**2})
            timings["total"] = (perf_counter() - start) * 1000
            # Validation stays outside timed regions; failures invalidate the run.
            if loss is not None and not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite loss")
            return timings

        memory_stages = [{"stage": "model_initialized", "allocated_mib": torch.cuda.memory_allocated() / 1024**2}]
        with torch.cuda.nvtx.range("warmup"):
            for _ in range(args.warmup):
                step()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        if args.memory_history:
            torch.cuda.memory._record_memory_history(max_entries=100000)
            history_started = True
        if args.profile_capture:
            torch.cuda.cudart().cudaProfilerStart()
            capture_started = True
        with torch.cuda.nvtx.range("measured"):
            samples = [step() for _ in range(args.iterations)]
        result["samples_ms"] = samples
        result["summary_ms"] = {
            phase: {"mean": statistics.mean(row[phase] for row in samples),
                    "stdev": statistics.stdev(row[phase] for row in samples) if len(samples) > 1 else 0.0}
            for phase in samples[0]
        }
        result["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
        result["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
        if args.memory_history:
            result["memory_stages"] = memory_stages
        result["status"] = "passed"
    except torch.OutOfMemoryError as error:
        result.update(status="oom", error=str(error))
    finally:
        if capture_started:
            torch.cuda.cudart().cudaProfilerStop()
        if history_started:
            args.memory_history.parent.mkdir(parents=True, exist_ok=True)
            torch.cuda.memory._dump_snapshot(str(args.memory_history))
            torch.cuda.memory._record_memory_history(enabled=None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] == "oom":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
