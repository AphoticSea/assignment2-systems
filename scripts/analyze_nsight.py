"""Derive phase/kernel findings directly from native Nsight CSV reports."""
import csv
import html
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/steps456/v3'


def read(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))


def table(headers,rows):
    return '<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</table>'


def main():
    records=[]
    for path in sorted((OUT/'profiles').glob('*_cuda_gpu_kern_sum.csv')):
        key=path.name.removesuffix('_cuda_gpu_kern_sum.csv')
        summary=OUT/'profiles'/f'{key}.json'
        if not summary.exists() or json.loads(summary.read_text())['status']!='passed':continue
        kernels=read(path)
        phases=read(OUT/'profiles'/f'{key}_nvtx_kern_sum.csv')
        projections=read(OUT/'profiles'/f'{key}_nvtx_gpu_proj_sum.csv')
        if not kernels or not phases:raise RuntimeError(f'No CUDA/NVTX data: {key}')
        phase_times={}
        phase_kernel_top={}
        for r in phases:
            phase=r['NVTX Range'].lstrip(':')
            phase_times[phase]=phase_times.get(phase,0)+float(r['Total Time (ns)'])/1e6
            if phase not in phase_kernel_top or float(r['Total Time (ns)'])>float(phase_kernel_top[phase]['Total Time (ns)']):phase_kernel_top[phase]=r
        total=sum(float(r['Total Time (ns)']) for r in kernels)
        matmul=sum(float(r['Total Time (ns)']) for r in kernels if re.search('gemm|gemv|cutlass|mma',r['Name'],re.I))
        measured=json.loads(summary.read_text())
        record=dict(case=key,kernel_time_ms=total/1e6,matmul_kernel_time_pct=100*matmul/total,
                    wall_ms=measured['summary_ms']['total']['mean'],phase_kernel_ms=phase_times,
                    phase_top_kernels=phase_kernel_top,
                    top_kernels=kernels[:10],projections=projections)
        base=OUT/'baseline'/f'{key}.json'
        record['baseline_wall_ms']=None
        if base.exists():
            baseline=json.loads(base.read_text())
            if baseline['status']=='passed':record['baseline_wall_ms']=baseline['summary_ms']['total']['mean']
        records.append(record)
    if not records:raise RuntimeError('No valid Nsight reports')
    (OUT/'nsight-analysis.json').write_text(json.dumps(records,indent=2)+'\n')
    selected=[r for r in records if r['case'] in ['small_ctx512_forward_fp32_w5','small_ctx512_train_fp32_w5']]
    fig,axes=plt.subplots(1,len(selected),figsize=(14,5),squeeze=False)
    for ax,r in zip(axes[0],selected):
        top=list(reversed(r['top_kernels'][:7]))
        labels=[]
        for k in top:
            name=k['Name']
            if 'cutlass' in name:name='cutlass SIMT SGEMM'
            elif 'magma_sgemm' in name:name='magma SGEMM'
            elif 'reduce' in name.lower():name='Reduction: '+name[:55]
            elif 'MulFunctor' in name:name='Elementwise multiply'
            elif 'DivFunctor' in name:name='Elementwise divide'
            elif 'where_kernel' in name:name='Elementwise where / mask'
            elif 'exp_kernel' in name:name='Elementwise exp'
            elif 'CUDAFunctor_add' in name:name='Elementwise add'
            else:name=name[:65]
            labels.append(name)
        ax.barh(range(len(top)),[float(k['Total Time (ns)'])/1e6 for k in top],color='#255dba')
        ax.set_yticks(range(len(top)),labels,fontsize=8)
        ax.set(xlabel='Cumulative GPU kernel time (ms)',title=r['case'].replace('_fp32_w5',''))
    fig.tight_layout();fig.savefig(OUT/'nsight-kernels.png',dpi=160);plt.close(fig)
    body='<h1>Nsight Systems：实际瓶颈与 Attention 优化空间</h1><p><a href="report.html">返回实验总报告</a></p><img src="nsight-kernels.png" style="width:100%">'
    body+='<p>下表的GPU时间是kernel duration之和；wall是包含CPU调度及CUDA同步的单步耗时。矩阵乘占比根据实际kernel名中的gemm/gemv/cutlass/mma识别，属于名称归类，不是硬件FLOP计数。嵌套NVTX range会重复包含kernel，不应将全部range相加。</p>'
    body+=table(['配置','profile wall ms','无profiler基线 ms','全部kernel ms','矩阵乘kernel时间占比'],[[r['case'],f"{r['wall_ms']:.2f}",f"{r['baseline_wall_ms']:.2f}" if r['baseline_wall_ms'] else '—',f"{r['kernel_time_ms']:.2f}",f"{r['matmul_kernel_time_pct']:.1f}%"] for r in records])
    for r in selected:
        body+='<h2>'+html.escape(r['case'])+'</h2>'
        body+=table(['NVTX阶段','累计GPU kernel时间 ms','最耗时kernel调用次数','最耗时kernel'],[[phase,f'{value:.3f}',r['phase_top_kernels'][phase]['Kern Inst'],r['phase_top_kernels'][phase]['Kernel Name']] for phase,value in r['phase_kernel_ms'].items() if phase!='measured'])
        soft=r['phase_kernel_ms'].get('attention_mask_softmax',0)
        qk=r['phase_kernel_ms'].get('attention_qk',0)
        pv=r['phase_kernel_ms'].get('attention_pv',0)
        body+=f'<p>Attention 前向阶段：QK（含scale）{qk:.3f}ms，mask/softmax {soft:.3f}ms，PV {pv:.3f}ms。这里只统计前向NVTX子阶段；反向kernel归入loss_backward。</p>'
    body+='<p>small、batch4、seq512、12 heads、head_dim64：每层的两次Attention矩阵乘合计约3.22×10⁹ FLOPs，按softmax每元素约5次基础操作估算约6.29×10⁷次（指数/归约不等价于一条普通FLOP）；12层分别再乘12。softmax的算术量明显较少，但需要读写大矩阵并启动多个kernel，因此时间比例不会按FLOPs比例缩小。</p>'
    body+='<p>优化建议：先用torch.compile检查这些elementwise/reduction操作能否融合，再实现FlashAttention减少分数矩阵物化、HBM读写和kernel启动。完整模型还包含投影与MLP的矩阵乘；必须同时报告算子加速与整模型加速。GPU kernel时间比例本身不是完整wall-clock速度提升的保证。</p>'
    style='body{font:16px/1.6 Segoe UI,Microsoft YaHei,sans-serif;max-width:1200px;margin:40px auto;padding:0 20px;color:#182238}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left;word-break:break-word}tr:nth-child(even){background:#f4f7fb}'
    (OUT/'nsight-findings.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Nsight瓶颈分析</title><style>'+style+'</style><body>'+body+'</body></html>',encoding='utf-8')
    for r in selected:print(r['case'],r['phase_kernel_ms'],'matmul_pct',r['matmul_kernel_time_pct'])


if __name__=='__main__':main()
