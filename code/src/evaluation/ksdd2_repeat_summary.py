"""Frozen Phase15D repeat policy and development summaries; blind test stays locked."""
import gc
import csv
import json
from pathlib import Path
import numpy as np
import torch
from src.evaluation import ksdd2_70k_protocol as parent
from src.evaluation.ksdd2_dev_evaluation import metrics,forward,paired,contamination_scan,KEYS

base=parent.base
OUT=base.ROOT/'experiments/phase15d_ksdd2_5seed_repeats'
SEEDS=(42,43,44,45,46)
NAMES=('STD','RECT')
RUNS=tuple((s,n) for s in SEEDS[1:] for n in NAMES)
FLAGS=dict(DEVELOPMENT_ONLY=True,DEV_NORMAL_REUSED_FOR_QUANTILE=True,REPEAT_COUNT=5,DGNB_ENABLED=False,
    DO_NOT_REPORT_AS_FINAL_KSDD2_TEST_RESULT=True,OFFICIAL_TEST_FORWARD_COUNT=0,
    OFFICIAL_TEST_PREDICTIONS_CREATED=0,FINAL_KSDD2_TEST_UNTOUCHED=True)
REFERENCE_SHA=dict(STD='62c708d75426b9352bb5cc434678faff2a90ea9203b987ffb06e28c37dbe0bec',
    RECT='b4428814ddbf83fde150353b1477ac9342e30962b7e56c6a1f12cf20bf371c41')

def key(seed,name):return f'{seed}_{name}'
def folder(seed,name):
    config(seed,name)
    return OUT/f'seed{seed}'/('std_256' if name=='STD' else 'rect_864x320')

def config(seed,name):
    if (seed,name) not in RUNS:raise ValueError('FAIL_NOT_REGISTERED_NEW_RUN')
    c=parent.config(name)
    c.update(FLAGS,stage='PHASE15D',seed=seed,seeds_recorded={k:seed for k in ('python','numpy','torch_cpu','torch_cuda')})
    if name=='RECT':c['method_description']='User-proposed Pixel-Budgeted Aspect-Ratio-Preserving EfficientAD-S / Ours'
    return c

def sequence(ids,seed):
    if seed not in SEEDS:raise ValueError('FAIL_SEED')
    order=np.random.default_rng(seed).permutation(len(ids))
    return [ids[int(order[i%len(ids)])] for i in range(70000)]

def checkpoint_gate(c,seed,name,final=False):
    if c.get('stage')!='PHASE15D' or c.get('seed')!=seed or c.get('method')!=name or c.get('config')!=config(seed,name):
        raise ValueError('FAIL_CROSS_RUN_OR_PILOT_CONTAMINATION')
    if c.get('input_hw')!=config(seed,name)['input_hw'] or c.get('parent_sha')!=base.PARENT_SHA or c.get('split_sha')!=base.SPLIT_SHA or not 0<=c.get('global_step',-1)<=70000:
        raise ValueError('FAIL_CHECKPOINT_BINDING')
    if final:
        q=c.get('quantiles',{})
        if c['global_step']!=70000 or c.get('validation_count')!=209 or set(q)!={'qa_st','qb_st','qa_ae','qb_ae'} or not all(np.isfinite(v) for v in q.values()):
            raise ValueError('FAIL_FINAL_VALIDATION')
        if any(q['qb_'+tag]<=q['qa_'+tag] for tag in ('st','ae')):raise ValueError('FAIL_QUANTILE_RANGE')
    return True

def healthy(r):
    return r.get('status')=='PASS' and r.get('completed')==70000 and all(r.get(k) for k in ('strict_reload','quantiles_reload','teacher_unchanged','student_updated','ae_updated')) and not any(r.get(k,True) for k in ('NaN','Inf','OOM'))

def all_ready(reports):
    if set(reports)!={key(s,n) for s,n in RUNS} or not all(healthy(r) for r in reports.values()):raise ValueError('FAIL_EIGHT_FINALS_REQUIRED')
    return True

def stats(values):
    a=np.asarray(values,dtype=np.float64)
    if len(a)!=5 or not np.isfinite(a).all():raise ValueError('FAIL_FIVE_FINITE_REPEATS')
    return dict(mean=float(a.mean()),std=float(a.std(ddof=1)),min=float(a.min()),max=float(a.max()),ddof=1)

def summarize_metrics(data):
    if set(data)!={str(s) for s in SEEDS}:raise ValueError('FAIL_FIVE_SEEDS')
    deltas={str(s):{k:100*(data[str(s)]['RECT'][k]-data[str(s)]['STD'][k]) for k in KEYS} for s in SEEDS}
    result={n:{k:stats([data[str(s)][n][k] for s in SEEDS]) for k in KEYS} for n in NAMES}
    delta={k:stats([deltas[str(s)][k] for s in SEEDS]) for k in KEYS}
    pro=sum(d['aupro005']>0 for d in deltas.values());pixel=sum(d['pixel_auroc']>0 for d in deltas.values())
    result.update(per_seed_delta_pp=deltas,delta_pp=delta,positive_aupro005_seeds=pro,positive_pixel_seeds=pixel,
        RECT_REPEAT_STABLE=pro>=4 and pixel>=4 and delta['aupro005']['mean']>3,
        RECT_REPEAT_STRONG=pro==5 and delta['aupro005']['mean']>=5-1e-12 and delta['pixel_auroc']['mean']>=1-1e-12 and delta['image_auroc']['mean']>=-2-1e-12)
    return result

def blind_protocol():
    return dict(dataset='KSDD2',seeds=list(SEEDS),baseline='Official EfficientAD-S baseline',baseline_input_hw=[256,256],
        ours='User-proposed Pixel-Budgeted Aspect-Ratio-Preserving EfficientAD-S',ours_input_hw=[864,320],
        steps=70000,batch=1,precision='FP32',image_score_protocol='official EfficientAD Combined',
        image_reduction='native Combined map maximum',pixel_map_protocol='official EfficientAD Combined',
        metrics=['Image AUROC','Pixel AUROC','AUPRO005','AUPRO030'],official_test_tuning_allowed=False,
        post_test_method_change_allowed=False,split_sha=base.SPLIT_SHA,DGNB_ENABLED=False)

def validate_reference_report(r):
    if r.get('status')!='PASS' or not r.get('FINAL_KSDD2_TEST_UNTOUCHED') or set(r.get('metrics',{}))!=set(NAMES):raise ValueError('FAIL_SEED42_DEV_REFERENCE')
    for n in NAMES:
        if r['training'][n]['checkpoint_sha256']!=REFERENCE_SHA[n] or not healthy(r['training'][n]):raise ValueError('FAIL_SEED42_DEV_REFERENCE')
        if not all(np.isfinite(r['metrics'][n][k]) and 0<=r['metrics'][n][k]<=1 for k in KEYS):raise ValueError('FAIL_SEED42_DEV_METRICS')
    return True

def verify_content_rows(rows):
    for r in rows:
        for kind in ('image','mask'):
            if base.parent.digest(r[kind+'_path'])!=r[kind+'_sha']:raise ValueError('FAIL_DEV_CONTENT_CHANGED '+r[kind+'_path'])
    return True

def development_assets(a,freeze=False):
    from src.training.ksdd2_formal_training import freeze_json
    reference=parent.OUT/'dev_evaluation/dev_report.json';mask_manifest=base.parent.OUT/'mask_manifest.csv'
    frozen=OUT/'development_asset_binding.json'
    if not freeze:
        expected=base.read(frozen)
        if base.parent.digest(reference)!=expected['seed42_dev_report_sha'] or base.parent.digest(mask_manifest)!=expected['mask_manifest_sha']:raise ValueError('FAIL_DEV_ASSET_CHANGED')
    validate_reference_report(base.read(reference))
    with mask_manifest.open(encoding='utf-8') as f:masks={r['id']:r for r in csv.DictReader(f)}
    rows=[]
    for r in a['groups']['validation_normal']+a['groups']['excluded_train_defect']:
        mr=masks[r['id']]
        if Path(mr['path']).resolve()!=Path(r['mask_path']).resolve():raise ValueError('FAIL_MASK_MAPPING')
        rows.append(dict(source_id=r['id'],image_path=r['image_path'],image_sha=r['image_sha'],mask_path=r['mask_path'],mask_sha=mr['sha']))
    value=dict(seed42_dev_report_sha=base.parent.digest(reference),mask_manifest_sha=base.parent.digest(mask_manifest),rows=rows)
    if freeze:freeze_json(frozen,value)
    elif value!=expected:raise ValueError('FAIL_DEV_BINDING')
    verify_content_rows(rows);return value

def evaluate_seed(seed,a,guard,reports):
    from src.training import ksdd2_repeat_training as t
    dest=OUT/f'seed{seed}'/'dev_evaluation';dest.mkdir(parents=True,exist_ok=True)
    result=dict(**FLAGS,seed=seed,status='RUNNING',metrics={},training=reports,state_unchanged={})
    images=[];regions=[];predictions=[];bounds=a['parent']['size_groups']['boundaries']
    for name in NAMES:
        m,c=t.load_final(seed,name);m.requires_grad_(False);before=t.old.state_hash(m);rows=[]
        nd=dest/'raw_native'/name.lower();nd.mkdir(parents=True,exist_ok=True)
        for i,r in enumerate(a['groups']['validation_normal']+a['groups']['excluded_train_defect']):
            x=t.old.normal_input(m,r['image_path'],guard,'dev');o=forward(m,x,name,tuple(config(seed,name)['input_hw']))
            hw=(int(r['height']),int(r['width']));pixel=base.native(o['pixel_map'],hw)
            path=nd/(r['id'].replace('/','__')+'.npy')
            if path.exists():
                if not np.array_equal(np.load(path,allow_pickle=False),pixel):raise ValueError('FAIL_EXISTING_MAP_CHANGED')
            else:np.save(path,pixel,allow_pickle=False)
            guard.check(r['mask_path'],'dev')
            rows.append(dict(source_id=r['id'],sample_id=r['id'],method=name,image_path=r['image_path'],gt_label=int(r['label']),
                gt_mask_path=r['mask_path'],native_shape=hw,raw_map_path=str(path),image_score=float(pixel.max()),
                size_group='ABCDE'[np.searchsorted(bounds,float(r['defect_ratio']),side='left')] if int(r['label']) else 'NORMAL'))
            del x,o,pixel
            if (i+1)%50==0:print(f'Seed{seed} {name} DEVELOPMENT {i+1}/455',flush=True)
        base.prediction_gate(a,rows)
        if before!=t.old.state_hash(m):raise ValueError('FAIL_INFERENCE_MUTATION')
        result['state_unchanged'][name]=True;del m,c;gc.collect();torch.cuda.empty_cache()
        base.csv_save(dest/f'{name.lower()}_dev_predictions.csv',rows)
        values,im,reg,meta=metrics(rows,name,dest/'metric_workspace'/name.lower())
        if meta['connected_regions']!=275 or len(meta['regions'])!=455:raise ValueError('FAIL_DEV_REGIONS')
        result['metrics'][name]=values;images+=im;regions+=reg;predictions+=rows
    for filename,rows in [('dev_image_metrics.csv',images),('dev_region_metrics.csv',regions),('dev_predictions.csv',predictions)]:base.csv_save(dest/filename,rows)
    groups=[]
    for name in NAMES:
        for g in 'ABCDE':
            v=[r['PRO005'] for r in images if r['method']==name and r['size_group']==g]
            groups.append(dict(method=name,size_group=g,n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),std=float(np.std(v))))
    result['size_groups_description_only']=groups;base.csv_save(dest/'dev_size_group_analysis.csv',groups)
    by={n:{r['source_id']:r['PRO005'] for r in images if r['method']==n} for n in NAMES};ids=sorted(by['STD'])
    if set(ids)!=set(by['RECT']):raise ValueError('FAIL_PAIRED_IDS')
    result['paired']=paired([by['RECT'][i] for i in ids],[by['STD'][i] for i in ids])
    by={n:{(r['source_id'],r['region_id']):r['PRO005'] for r in regions if r['method']==n} for n in NAMES};s,r=by['STD'],by['RECT']
    if set(s)!=set(r):raise ValueError('FAIL_REGION_IDS')
    result['region_paired']=dict(RECT_ONLY=sum(r[k]>0 and s[k]<=0 for k in s),STD_ONLY=sum(r[k]<=0 and s[k]>0 for k in s),BOTH=sum(r[k]>0 and s[k]>0 for k in s),BOTH_BAD=sum(r[k]<=0 and s[k]<=0 for k in s))
    result.update(contamination_scan(a,dest));result.update(status='PASS',access=guard.report(),Approximation=False)
    base.save(dest/'dev_report.json',result);return result

def final_text(r):
    s=r['summary'];lines=['='*40,'PHASE 15D - KSDD2 5-SEED REPEATS','='*40,'Seeds: 42,43,44,45,46',
        'Official test untouched: '+str(r['FINAL_KSDD2_TEST_UNTOUCHED']),'-'*40,'CHECKPOINTS']
    for seed in SEEDS:
        for n in NAMES:
            run=r['training'][str(seed)][n]
            lines += [f'Seed{seed} {n}: PASS',f'SHA: {run["checkpoint_sha256"]}',
                f'Checkpoint: {run["checkpoint"]}',f'Training hours: {run["training_hours"]:.8f}',
                'Quantiles: '+json.dumps(run['quantiles'])]
    lines += ['-'*40,'5-SEED DEV RESULTS (percent; sample std ddof=1)']
    for n in NAMES:
        lines.append(n+':')
        for k in KEYS:lines.append(f'{k}: {100*s[n][k]["mean"]:.8f} +/- {100*s[n][k]["std"]:.8f}')
    lines += ['-'*40,'RECT - STD (pp)']
    for k in KEYS:lines.append(f'{k}: {s["delta_pp"][k]["mean"]:.8f} +/- {s["delta_pp"][k]["std"]:.8f}')
    lines += [f'Positive AUPRO005 seeds: {s["positive_aupro005_seeds"]}/5',f'Positive Pixel seeds: {s["positive_pixel_seeds"]}/5',
        f'RECT_REPEAT_STABLE: {s["RECT_REPEAT_STABLE"]}',f'RECT_REPEAT_STRONG: {s["RECT_REPEAT_STRONG"]}',
        '-'*40,'TRAINING COST',json.dumps(r['training_cost'],indent=2),'-'*40,'BLIND TEST',
        'FINAL_BLIND_TEST_PROTOCOL: FROZEN','Official test forward count: 0','Official test predictions: 0',
        'FINAL_KSDD2_TEST_UNTOUCHED: True',f'READY_FOR_5SEED_ONE_SHOT_TEST: {r["READY_FOR_5SEED_ONE_SHOT_TEST"]}',
        'STATUS: '+r['status'],'='*40]
    return '\n'.join(lines)

def run(a,guard,reports,references):
    from src.training import ksdd2_repeat_training as t
    all_ready(reports)
    # Verify every checkpoint before permitting the first new development forward.
    for s,n in RUNS:
        r=reports[key(s,n)]
        if base.parent.digest(r['checkpoint'])!=r['checkpoint_sha256']:raise ValueError('FAIL_SHA')
        m,c=t.load_final(s,n);del m,c;gc.collect();torch.cuda.empty_cache()
    reference=t.references()
    if reference!=references:raise ValueError('FAIL_SEED42_REFERENCE')
    development_assets(a)
    guard.ready=True
    data={'42':base.read(parent.OUT/'dev_evaluation/dev_report.json')}
    for s in SEEDS[1:]:data[str(s)]=evaluate_seed(s,a,guard,{n:reports[key(s,n)] for n in NAMES})
    summary=summarize_metrics({s:r['metrics'] for s,r in data.items()})
    training={'42':references,**{str(s):{n:reports[key(s,n)] for n in NAMES} for s in SEEDS[1:]}}
    costs={n:{k:stats([training[str(s)][n][k] for s in SEEDS]) for k in ('training_hours','peak_allocated_mib','peak_reserved_mib')} for n in NAMES}
    costs['RECT_STD_training_ratio']=costs['RECT']['training_hours']['mean']/costs['STD']['training_hours']['mean']
    groups=[]
    for n in NAMES:
        for g in 'ABCDE':
            rows=[next(x for x in data[str(s)]['size_groups_description_only'] if x['method']==n and x['size_group']==g) for s in SEEDS]
            groups.append(dict(method=n,size_group=g,n=rows[0]['n'],**{k:stats([r[k] for r in rows]) for k in ('mean','median','std')}))
    scan=contamination_scan(a,OUT)
    ready=all(healthy(r) for runs in training.values() for r in runs.values()) and not guard.blocked and scan['FINAL_KSDD2_TEST_UNTOUCHED']
    if base.assets()['split_sha']!=base.SPLIT_SHA:raise ValueError('FAIL_SPLIT')
    result=dict(**FLAGS,status='PASS',metrics_by_seed={s:r['metrics'] for s,r in data.items()},summary=summary,training=training,
        training_cost=costs,size_groups_description_only=groups,paired_by_seed={s:r['paired'] for s,r in data.items()},
        regions_by_seed={s:r['region_paired'] for s,r in data.items()},access=guard.report(),contamination=scan,
        READY_FOR_5SEED_ONE_SHOT_TEST=bool(ready),Approximation=False)
    dest=OUT/'summary';t.core.freeze_json(dest/'FINAL_BLIND_TEST_PROTOCOL.json',blind_protocol())
    base.save(dest/'repeat_report.json',result)
    base.csv_save(dest/'per_seed_metrics.csv',[dict(seed=s,method=n,**data[str(s)]['metrics'][n]) for s in SEEDS for n in NAMES])
    base.csv_save(dest/'per_seed_delta.csv',[dict(seed=s,**summary['per_seed_delta_pp'][str(s)]) for s in SEEDS])
    text=final_text(result)+'\n\nFULL PRECISION RECORD\n'+json.dumps(result,indent=2)
    (dest/'repeat_report.txt').write_text(text,encoding='utf-8');print(text,flush=True);return result
