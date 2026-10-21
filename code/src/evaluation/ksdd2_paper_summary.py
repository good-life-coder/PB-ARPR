"""Frozen Phase15E descriptive summaries; no selection or training."""
import numpy as np
from src.evaluation import ksdd2_repeat_summary as repeat

KEYS=repeat.KEYS

def summarize(data):
    r=repeat.summarize_metrics(data)
    r.pop('RECT_REPEAT_STABLE');r.pop('RECT_REPEAT_STRONG')
    d=r['delta_pp']
    r.update(OFFICIAL_RECT_SUCCESS=d['aupro005']['mean']>0 and d['pixel_auroc']['mean']>0 and d['image_auroc']['mean']>=-2-1e-12,
        OFFICIAL_RECT_STRONG=r['positive_aupro005_seeds']==5 and d['aupro005']['mean']>=3-1e-12 and r['positive_pixel_seeds']>=4 and d['image_auroc']['mean']>=-2-1e-12,
        DEV_TO_TEST_DIRECTION_PERSISTED=d['aupro005']['mean']>0 and d['pixel_auroc']['mean']>0 and d['image_auroc']['mean']>-2)
    return r

def display(v,scale=100):return f"{scale*v['mean']:.8f} +/- {scale*v['std']:.8f}"

def paper_rows(summary,runtime):
    return [dict(Method='Official EfficientAD-S' if n=='STD' else 'Ours (PB-ARPR EfficientAD-S)',
        **{k:display(summary[n][k]) for k in KEYS},Params=8058628,Input='256x256' if n=='STD' else '864x320',
        Runtime_ms=runtime[n]['model_only']['mean_ms'],VRAM_allocated_MiB=runtime[n]['peak_allocated_mib']) for n in repeat.NAMES]

def aggregate_groups(by_seed):
    rows=[]
    for g in 'ABCDE':
        pairs={n:[next(x for x in by_seed[str(seed)] if x['method']==n and x['size_group']==g) for seed in repeat.SEEDS] for n in repeat.NAMES}
        rows.append(dict(size_group=g,n=pairs['STD'][0]['n'],
            **{n:repeat.stats([r['mean'] for r in pairs[n]]) for n in repeat.NAMES},
            delta_pp=repeat.stats([100*(r['mean']-s['mean']) for r,s in zip(pairs['RECT'],pairs['STD'])])))
    return rows

def text_report(r):
    lines=['='*40,'PHASE 15E - KSDD2 BLIND OFFICIAL TEST','='*40,
        'FIRST OFFICIAL TEST: True','Seeds: 42,43,44,45,46','Test: 1004; Normal: 894; Defect: 110',
        'PER-SEED RESULTS (%)']
    for seed,methods in r['metrics_by_seed'].items():
        for name,values in methods.items():lines.append(f'Seed{seed} {name}: '+', '.join(f'{k}={100*values[k]:.8f}' for k in KEYS))
    lines+=['5-SEED MEAN +/- SAMPLE STD (%)']
    for n in repeat.NAMES:
        lines += [n]+[f'{k}: {display(r["summary"][n][k])}' for k in KEYS]
    lines+=['RECT - STD (pp)']+[f'{k}: {display(r["summary"]["delta_pp"][k],1)}' for k in KEYS]
    lines += ['DECISION',str({k:v for k,v in r['summary'].items() if isinstance(v,(bool,int))}),
        'METHOD_FROZEN_BEFORE_TEST: True','OFFICIAL_TEST_USED_FOR_METHOD_SELECTION: False','NO_POST_TEST_TUNING: True',
        'POST_TEST_TRAINING_STARTED: False','STATUS: '+r['status'],'='*40]
    return '\n'.join(lines)
