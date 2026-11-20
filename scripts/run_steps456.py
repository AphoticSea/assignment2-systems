"""Run isolated, resumable experiments for 2025 A2 sections 1.1.3--1.1.6."""
import argparse
import hashlib
import json
import os
import signal
import shutil
from pathlib import Path
import subprocess
import sys
import time

import torch
from cs336_basics.model import BasicsTransformerLM
from cs336_systems.benchmark import SIZES

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_HASH=hashlib.sha256((ROOT/'cs336_systems/benchmark.py').read_bytes()).hexdigest()
OUT = ROOT / "results/steps456" / os.environ.get("CS336_RUN_NAME", "v3")
NSYS = Path(os.environ.get('NSYS_BIN') or shutil.which('nsys') or str(Path.home() / ".local/share/cs336/nsight/extracted/opt/nvidia/nsight-systems-cli/2026.6.1/target-linux-x64/nsys"))


def counts():
    result = {}
    with torch.device("meta"):
        for size, (d, f, layers, heads) in SIZES.items():
            model = BasicsTransformerLM(10000, 128, d, layers, heads, f, 10000.0)
            result[size] = sum(p.numel() for p in model.parameters())
    return result


def run_case(group, size, context, mode, dtype="fp32", warmup=5, iterations=10, snapshot=False, profile=False):
    folder = OUT / group
    folder.mkdir(parents=True, exist_ok=True)
    key = f"{size}_ctx{context}_{mode}_{dtype}_w{warmup}"
    target = folder / (key + ".json")
    if target.exists():
        old = json.loads(target.read_text())
        if old.get("gpu") == GPU_NAME and old.get('benchmark_sha256') == BENCHMARK_HASH and old.get("status") in {"passed", "oom", "capacity_excluded"} and (not profile or old["status"] != "passed" or (folder / (key + ".nsys-rep")).exists()):
            print("cached", group, key, old["status"], flush=True)
            return old
    params = PARAMETERS[size]
    lower = params * 4 * {"forward": 1, "forward-backward": 2, "train": 4}[mode]
    symbolic = CAPACITY[f"{size}_ctx{context}_{dtype}"]
    forward_bound = symbolic["forward_lower_bound_bytes"]
    if mode == "train":
        forward_bound += params * 8  # Adam's two FP32 moment buffers after warmup.
    lower = max(lower, forward_bound)
    if lower > FREE_BYTES:
        result = dict(status="capacity_excluded", size=size, context_length=context, mode=mode,
                      dtype=dtype, gpu=GPU_NAME, benchmark_sha256=BENCHMARK_HASH, parameters=params, minimum_resident_bytes=lower,
                      available_reference_bytes=FREE_BYTES,
                      reason="Symbolic saved-activation / FP32 state lower bound exceeds 90% allocator budget; not an observed OOM")
        target.write_text(json.dumps(result, indent=2) + "\n")
        print(group, key, "capacity_excluded", flush=True)
        return result
    cmd = [sys.executable, "-m", "cs336_systems.benchmark", "--size", size,
           "--context-length", str(context), "--mode", mode, "--dtype", dtype,
           "--warmup", str(warmup), "--iterations", str(iterations), "--output", str(target)]
    if snapshot:
        cmd += ["--memory-history", str(folder / (key + ".pickle"))]
    if profile:
        cmd += ["--profile-capture", "--annotate-attention"]
        cmd = [str(NSYS), "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
               "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "--force-overwrite=true",
               "--output", str(folder / key)] + cmd
    started = time.time()
    with (folder / (key + ".log")).open("w") as log:
        try:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True, env={**os.environ, "WANDB_MODE": "disabled"})
            code = proc.wait(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            code = "timeout"
    if not target.exists():
        result = dict(status="timeout" if code == "timeout" else "error", returncode=code, command=cmd)
        target.write_text(json.dumps(result, indent=2) + "\n")
    result = json.loads(target.read_text())
    print(group, key, result["status"], f"{time.time()-started:.1f}s", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["baseline", "profiles", "memory"])
    args = parser.parse_args()
    global PARAMETERS, FREE_BYTES, CAPACITY, GPU_NAME
    PARAMETERS = counts()
    probe=json.loads(subprocess.check_output([sys.executable,'-c',
        'import json,torch; f,t=torch.cuda.mem_get_info(); print(json.dumps(dict(free=f,total=t,gpu=torch.cuda.get_device_name())))'],text=True))
    GPU_NAME=probe['gpu']
    FREE_BYTES = min(probe['free'], int(probe['total']*0.90))
    CAPACITY = json.loads((OUT.parent / "capacity.json").read_text())
    print("Parameter counts:", PARAMETERS, flush=True)
    if args.phase == "baseline":
        for dtype in ["fp32", "bf16"]:
            for size in list(SIZES)[1:]:
                for context in [128, 256, 512, 1024]:
                    for mode in ["forward", "forward-backward"]:
                        run_case("baseline", size, context, mode, dtype)
        for size in list(SIZES)[1:]:
            for warmup in [0, 1, 2]:
                for mode in ["forward", "forward-backward"]:
                    run_case("warmup", size, 128, mode, warmup=warmup)
    elif args.phase == "profiles":
        for size in list(SIZES)[1:]:
            for context in [128, 256, 512, 1024]:
                for mode in ["forward", "forward-backward", "train"]:
                    run_case("profiles", size, context, mode, iterations=1, profile=True)
    else:
        for size in ["2.7B", "small"]:
            for dtype in ["fp32", "bf16"]:
                for context in [128, 256, 512]:
                    for mode in ["forward", "train"]:
                        run_case("memory", size, context, mode, dtype, warmup=2, iterations=1, snapshot=True)


if __name__ == "__main__":
    main()
