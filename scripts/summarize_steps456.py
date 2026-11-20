"""Generate a local report and figures from measured results, never invented timings."""
import csv
import html
import json
from pathlib import Path
import pickle
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.cuda._memory_viz as memory_viz

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/steps456/v3"
PATTERN=re.compile(r"^(small|medium|large|xl|2\.7B)_ctx(\d+)_(forward-backward|forward|train)_(fp32|bf16)_w(\d+)$")


def load(group):
    rows=[]
    for path in sorted((OUT/group).glob("*.json")):
        match=PATTERN.match(path.stem)
        if not match:
            continue
        r=json.loads(path.read_text())
        r.update(size=match[1],context=int(match[2]),mode=match[3],dtype=match[4],warmup=int(match[5]),file=path.name)
        rows.append(r)
    return rows


def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table>'


def result_table(rows):
    data=[]
    for r in rows:
        if r['status']=='passed':
            t=r['summary_ms']['total']
            values=[f"{t['mean']:.2f} ± {t['stdev']:.2f}",f"{r['peak_allocated_mib']:.1f}"]
        else:
            values=['—','—']
        data.append([r['size'],r['context'],r['mode'],r['dtype'],r['warmup'],r['status'],*values])
    return table(['配置','序列长度','模式','精度','预热','结果','耗时 ms（均值 ± 标准差）','峰值 allocated MiB'],data)


def charts(baseline,warmups,memory):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(13,3.6))
    for ax,size in zip(axes,['small','medium','large']):
        for dtype,color in [('fp32','#255dba'),('bf16','#ca6c20')]:
            rows=sorted([r for r in baseline if r['size']==size and r['mode']=='forward-backward' and r['dtype']==dtype and r['status']=='passed'],key=lambda r:r['context'])
            ax.plot([r['context'] for r in rows],[r['summary_ms']['total']['mean'] for r in rows],'-o',label=dtype,color=color)
        ax.set(title=size,xlabel='Sequence length',ylabel='Forward + loss + backward (ms)',xticks=[128,256,512,1024])
        ax.set_xscale('log',base=2)
        ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(OUT/'baseline.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,3.6))
    for ax,size in zip(axes,['small','medium','large']):
        rows=sorted([r for r in warmups+baseline if r['size']==size and r['context']==128 and r['mode']=='forward' and r['dtype']=='fp32' and r['status']=='passed'],key=lambda r:r['warmup'])
        ax.plot([r['warmup'] for r in rows],[r['summary_ms']['total']['mean'] for r in rows],'-o',label='10-sample mean')
        ax.plot([r['warmup'] for r in rows],[r['samples_ms'][0]['total'] for r in rows],'--s',label='First measured sample')
        ax.set(title=size,xlabel='Explicit model warmup steps',ylabel='Forward wall time (ms)',xticks=[0,1,2,5]);ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(OUT/'warmup.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for mode,marker in [('forward','o'),('train','s')]:
        for dtype,color in [('fp32','#255dba'),('bf16','#ca6c20')]:
            rows=sorted([r for r in memory if r['size']=='small' and r['mode']==mode and r['dtype']==dtype and r['status']=='passed'],key=lambda r:r['context'])
            ax.plot([r['context'] for r in rows],[r['peak_allocated_mib']/1024 for r in rows],marker=marker,linestyle='-' if mode=='train' else '--',color=color,label=f'{mode}, {dtype}')
    ax.set(xlabel='Sequence length',ylabel='Peak allocated GPU memory (GiB)',title='Local small-model memory profiles (FP32 master weights)');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'memory.png',dpi=160);plt.close(fig)


def snapshots():
    fig,axes=plt.subplots(1,3,figsize=(14,3.8))
    selected=[(OUT/'memory/small_ctx512_forward_fp32_w2.pickle','Training forward (grad enabled)'),
              (OUT/'memory/small_ctx512_train_fp32_w2.pickle','Full AdamW training step'),
              (OUT/'inference/small_ctx512_fp32.pickle','Inference (no grad)')]
    summary={}
    for ax,(path,title) in zip(axes,selected):
        with path.open('rb') as f:
            snapshot=pickle.load(f)
        # Native PyTorch memory-viz HTML; snapshots are created by this task.
        path.with_suffix('.html').write_text(memory_viz.trace_plot(snapshot,device=0),encoding='utf-8')
        events=snapshot['device_traces'][0]
        active_end=sum(block['size'] for segment in snapshot['segments'] for block in segment['blocks'] if block['state'].startswith('active'))
        net=sum(event['size']*(1 if event['action']=='alloc' else -1) for event in events if event['action'] in {'alloc','free_requested'})
        value=active_end-net
        x=[0]; y=[value/1024**3]
        for i,event in enumerate(events):
            if event['action']=='alloc':value+=event['size']
            elif event['action']=='free_requested':value-=event['size']
            else:continue
            x.append(i+1);y.append(value/1024**3)
        ax.step(x,y,where='post',color='#255dba')
        ax.set(title=title,xlabel='Allocator event index (not wall time)',ylabel='Active block memory (GiB)');ax.grid(alpha=.2)
        allocations=sorted([e for e in events if e['action']=='alloc'],key=lambda e:e['size'],reverse=True)
        summary[path.stem]={'initial_active_gib':y[0],'peak_active_gib':max(y),'event_count':len(events),
                           'largest_allocations':[{'mib':e['size']/1024**2,'frames':e.get('frames',[])[:5]} for e in allocations[:6]]}
    fig.tight_layout();fig.savefig(OUT/'memory-timelines.png',dpi=160);plt.close(fig)
    (OUT/'snapshot-analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def nsight_analysis():
    results=[]
    for path in sorted((OUT/'profiles').glob('*_nvtx_kern_sum.csv')):
        with path.open(newline='') as f:
            rows=list(csv.DictReader(f))
        if not rows:continue
        results.append({'file':path.name,'columns':list(rows[0]),'rows':rows})
    (OUT/'nsight-tables.json').write_text(json.dumps(results,indent=2)+'\n')
    return results


def main():
    groups={g:load(g) for g in ['baseline','warmup','profiles','memory']}
    expected={'baseline':80,'warmup':30,'profiles':60,'memory':24}
    for group,rows in groups.items():
        assert len(rows)==expected[group],(group,len(rows))
        assert not [r for r in rows if r['status'] not in {'passed','oom','capacity_excluded'}],group
    charts(groups['baseline'],groups['warmup'],groups['memory'])
    snapshot_summary=snapshots()
    nsight=nsight_analysis()
    assert nsight,'No usable native Nsight CUDA/NVTX tables'
    probe=json.loads((OUT.parent/'mixed_precision_probe.json').read_text())
    assert probe['small_model_numerics']['all_bf16_gradients_finite']
    env=json.loads((ROOT/'results/environment.json').read_text())
    counts={g:{s:sum(r['status']==s for r in rows) for s in ['passed','oom','capacity_excluded']} for g,rows in groups.items()}
    (OUT/'completion.json').write_text(json.dumps({'counts':counts,'environment':env,'nsight_tables':len(nsight)},indent=2)+'\n')
    body='<h1>CS336 2025 Systems：步骤 4–6 实验报告</h1>'
    body+='<p>GPU：RTX 5070 Ti Laptop（约12GB）。环境：CS336 / WSL2 Ubuntu 22.04 / Python '+env['python']+' / torch '+env['torch']+' / Triton '+env['triton']+'。</p>'
    body+='<p>课程对应：§1.1.2–1.1.3 基线与预热；§1.1.4 Nsight；§1.1.5 混合精度；§1.1.6 显存。原始模型源码和课程测试保持不变。随机数据用于性能实验，不代表训练收敛结果。</p>'
    body+=table(['实验组','实际测得','真实 allocator OOM','容量预检排除'],[[g,*[counts[g][s] for s in ['passed','oom','capacity_excluded']]] for g in counts])
    body+='<p>本轮统一将 PyTorch allocator 限制为显存的90%，避免 Windows/WSL 显存回退。capacity_excluded 是根据 FP32 参数/梯度/Adam 状态和 meta 张量保留激活计算的下限，未实际运行；oom 是实际抛出的 allocator 异常。BF16 的容量预检使用更保守的全BF16符号激活下限，实际混合精度可能占用更多显存。</p>'
    body+='<p>时间采用 perf_counter 和阶段间 CUDA 同步，包含主机调度与同步开销，zero_grad 不计入。前向计时默认保留 autograd，因此与 no_grad 推理不同。初始化在GPU上完成，loss以FP32计算，TF32关闭。正式矩阵预热5次、计时10次；profile 只采集预热后的1步；显存 snapshot 预热2次后采集1步。旧 protocol2 的部分试跑保留在上级目录，本文仅使用v3。</p>'
    body+='<h2>4. 基线与预热</h2><img src="baseline.png"><img src="warmup.png">'
    body+='<p>预热消融使用全新进程，比较0、1、2次预热与基线5次预热。首次CUDA/cuBLAS/算子路径初始化和主机开销可显著抬高首个样本；不能把冷启动均值与稳态性能混在一起。</p>'
    body+='<details><summary>完整80项基线矩阵</summary>'+result_table(groups['baseline'])+'</details>'
    body+='<details><summary>完整30项预热消融</summary>'+result_table(groups['warmup'])+'</details>'
    body+='<h2>5. Nsight Systems</h2><p>Nsight Systems CLI 2026.6.1；trace=cuda,nvtx；关闭CPU采样。cudaProfilerStart/Stop 排除模型初始化和预热；forward、loss_backward、optimizer，以及 attention_qk / attention_mask_softmax / attention_pv 由NVTX标注。原生 .nsys-rep、SQLite 与按range汇总的CSV均保留在 profiles/。NVTX嵌套range会重复包含同一个kernel，不能将所有range简单相加。</p>'
    body+='<p>有效NVTX kernel汇总表：'+str(len(nsight))+'。详细结论见 <a href="nsight-findings.html">Nsight瓶颈分析</a>。</p>'
    body+='<details><summary>全部profile配置与测量状态</summary>'+result_table(groups['profiles'])+'</details>'
    body+='<h2>6. 混合精度与显存</h2><img src="memory.png"><img src="memory-timelines.png">'
    body+='<p>内存曲线来自真实 PyTorch allocator history。横轴为分配事件序号而非墙钟时间；原生 memory-viz HTML 和 .pickle 同时保留。图中训练步已经完成预热，因此Adam状态在初始基线内。PyTorch snapshot 不包含CUDA上下文等allocator之外的显存。</p>'
    body+='<p>原题2.7B配置在随附实现中实为3,406,809,600个参数，FP32参数下限12.69GiB；FP32参数、梯度和两个Adam状态的下限50.76GiB，均未包含激活。因此该配置在本机被容量排除，§1.1.6的2.7B原题图与数值仍需大显存GPU完成。small的snapshot属于明确标记的本地适配实验。</p>'
    body+='<p>2.7B配置的单个FP32残差张量大小为 batch×seq×d_model×4 bytes；batch=4、d_model=2560时，长度128/256/512分别为5/10/20MiB。</p>'
    body+='<details><summary>全部显存实验状态</summary>'+result_table(groups['memory'])+'</details>'
    body+='<h3>累加精度实验</h3>'+table(['设置','累加结果','相对真实值10的绝对误差'],[[name,v['value'],v['absolute_error']] for name,v in probe['accumulation'].items()])
    body+='<h3>Autocast dtype实测</h3><pre>'+html.escape(json.dumps(probe['autocast'],indent=2,ensure_ascii=False))+'</pre>'
    body+='<p>FP32累加可减轻重复舍入误差；将已量化的FP16输入再转回FP32，无法恢复输入量化时丢失的精度。LayerNorm的均值/方差归约对精度敏感；BF16保留FP32的指数范围，但尾数更短，不能仅凭指数范围就判断所有归约都适合低精度。</p>'
    body+='<h3>同权重small模型的数值检查</h3><pre>'+html.escape(json.dumps(probe['small_model_numerics'],indent=2))+'</pre>'
    body+='<p>该检查比较同一组FP32权重、输入和目标下的FP32与BF16输出、loss和第一层Q投影梯度；它不构成长期训练收敛保证。</p>'
    body+='<h2>复现与后续</h2><pre>conda activate CS336\nbash scripts/run_steps456.sh\npython scripts/inference_memory.py\npython scripts/export_nsight.py\npython scripts/summarize_steps456.py</pre>'
    body+='<p>下一步：§1.3 的 torch.compile（原题先有§1.2.1 Attention基准），随后§1.3.1 Triton、§1.3.2 FlashAttention前向及反向。租卡比较必须在每张GPU内部计算优化加速比，保持输入、精度、软件、计时协议和allocator预算一致。2.7B完整训练建议80GB级GPU，先用同一容量预检确认目标形状能否放下。</p>'
    body+='<p>资料：<a href="../../../cs336_spring2025_assignment2_systems.pdf">2025官方手册</a>；<a href="https://docs.pytorch.org/docs/stable/torch_cuda_memory">PyTorch memory snapshot说明</a>；<a href="https://docs.nvidia.com/nsight-systems/InstallationGuide/">Nsight官方文档</a>。</p>'
    css='body{font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif;max-width:1120px;margin:40px auto;padding:0 24px;color:#172334}h1,h2{line-height:1.3}table{border-collapse:collapse;width:100%;font-size:13px;margin:18px 0}td,th{padding:8px;border-bottom:1px solid #dbe1e8;text-align:left}th{background:#edf2f8}tr:nth-child(even){background:#f7f9fc}img{width:100%;height:auto;margin:14px 0}pre{white-space:pre-wrap;background:#edf2f8;padding:16px;font-size:13px}details{margin:18px 0}summary{cursor:pointer;font-weight:600}'
    (OUT/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>CS336 Systems 步骤4–6</title><style>'+css+'</style><body>'+body+'</body></html>',encoding='utf-8')
    print(json.dumps(counts,indent=2));print(OUT/'report.html')


if __name__=='__main__':
    main()
