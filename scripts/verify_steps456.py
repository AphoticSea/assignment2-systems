"""Validate completeness, provenance and numerical checks of the experiment artifacts."""
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/steps456'/os.environ.get('CS336_RUN_NAME','v3')


def main():
    counts={}
    hashes=set()
    for group,expected in [('baseline',80),('warmup',30),('profiles',60),('memory',24)]:
        rows=[(p,json.loads(p.read_text())) for p in (OUT/group).glob('*.json')]
        assert len(rows)==expected,(group,len(rows),expected)
        counts[group]={}
        for path,r in rows:
            assert r['status'] in {'passed','oom','capacity_excluded'},(path,r)
            counts[group][r['status']]=counts[group].get(r['status'],0)+1
            if r['status']=='passed':
                assert r['protocol_version']==3 and r['allocator_memory_fraction']==0.90
                assert len(r['samples_ms'])==r['iterations']
                assert r['summary_ms']['total']['mean']>0
                assert r['peak_allocated_mib']>0
                hashes.add(r['benchmark_sha256'])
                if group=='profiles':assert path.with_suffix('.nsys-rep').exists()
                if group=='memory':assert path.with_suffix('.pickle').exists()
    assert len(hashes)==1,hashes
    probe=json.loads((OUT/'mixed_precision_probe.json').read_text())
    numerics=probe['small_model_numerics']
    assert numerics['all_bf16_gradients_finite']
    assert numerics['logits_relative_l2']<0.05
    assert numerics['first_q_proj_gradient_relative_l2']<0.05
    nsight=json.loads((OUT/'nsight-analysis.json').read_text())
    assert len(nsight)==counts['profiles']['passed']
    for row in nsight:
        assert abs(row['phase_kernel_ms']['measured']-row['kernel_time_ms'])<0.1,row['case']
    subprocess.run(['git','diff','--exit-code','--','tests','cs336-basics/cs336_basics'],cwd=ROOT,check=True)
    result={'status':'passed','matrix_counts':counts,'unique_benchmark_hashes':list(hashes),
            'native_nsight_traces_with_valid_kernel_data':len(nsight),
            'staff_model_and_original_tests':'unchanged',
            'mixed_precision_logit_and_selected_gradient_relative_l2_tolerance':0.05}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
