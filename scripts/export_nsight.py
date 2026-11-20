"""Export native Nsight reports; retain .nsys-rep and SQLite for inspection."""
from pathlib import Path
import json
import os
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
NSYS = Path(os.environ.get('NSYS_BIN') or shutil.which('nsys') or str(Path.home()/".local/share/cs336/nsight/extracted/opt/nvidia/nsight-systems-cli/2026.6.1/target-linux-x64/nsys"))
FOLDER = ROOT/'results/steps456'/os.environ.get('CS336_RUN_NAME','v3')/'profiles'


def main():
    for trace in sorted(FOLDER.glob("*.nsys-rep")):
        case = trace.with_suffix('.json')
        if not case.exists() or json.loads(case.read_text()).get('status') != 'passed':
            continue
        output = trace.with_suffix("")
        reports = ["cuda_gpu_kern_sum", "nvtx_kern_sum", "nvtx_gpu_proj_sum"]
        expected = [Path(str(output)+"_"+r+".csv") for r in reports]
        if all(p.exists() and p.stat().st_size > 0 for p in expected):
            continue
        with Path(str(output)+"_stats.log").open("w") as log:
            proc = subprocess.run([str(NSYS),"stats","--report",",".join(reports),
                                   "--format","csv","--force-export=true","--force-overwrite=true",
                                   "--output",str(output),str(trace)],stdout=log,stderr=subprocess.STDOUT)
        print(trace.name, "exported" if proc.returncode == 0 else f"failed({proc.returncode})", flush=True)


if __name__ == "__main__":
    main()
