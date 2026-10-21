"""Phase15C development-only adapter; all metric math reused from Phase15B."""
import gc
import json
from pathlib import Path
import numpy as np
import torch
from src.evaluation import ksdd2_70k_protocol as p
from src.evaluation.ksdd2_dev_evaluation import metrics,forward,paired,contamination_scan,KEYS
from src.training import ksdd2_formal_training as training

def final_text(r):
    lines=['='*40,'PHASE 15C - KSDD2 70K STD VS RECT','='*40,'Protocol: Development verification before blind test','Official test untouched: True']
    for n in p.NAMES:
        t=r['training'][n]
        lines+=['-'*40,n+' 70K','Input: '+'x'.join(map(str,t['input_hw'])),'Method: '+t['method_description'],
            f'Completed: {t["completed"]}/70000',f'Training hours: {t["training_hours"]:.6f}',f'Seconds/step: {t["seconds_per_step"]:.6f}',
            f'Peak VRAM allocated/reserved: {t["peak_allocated_mib"]:.3f}/{t["peak_reserved_mib"]:.3f} MiB',
            'Checkpoint: '+t['checkpoint'],'SHA256: '+t['checkpoint_sha256'],'Strict reload: PASS',
            'Quantiles: '+json.dumps(t['quantiles']),'Resume used: '+str(t['resume_used'])]
    lines+=['-'*40,'70K DEV PERFORMANCE (percent)','Metric                 STD           RECT']
    for k in KEYS:lines.append(k.ljust(20)+' '.join(f'{r["metrics"][n][k]*100:12.6f}' for n in p.NAMES))
    lines+=['-'*40,'10K VS 70K DELTAS / DECISION',json.dumps(r['decision'],indent=2),
        'PAIRED: '+json.dumps(r['paired']), 'REGIONS: '+json.dumps(r['region_paired']),
        'REPEAT_COUNT: 1','PAPER_FINAL_REQUIRES_REPEAT_ANALYSIS: True','DEV_NORMAL_REUSED_FOR_QUANTILE: True',
        'Official test forward count: 0','Official test predictions: 0','FINAL_KSDD2_TEST_UNTOUCHED: True','STATUS: PASS','='*40]
    return '\n'.join(lines)

def run(a,guard,reports):
    p.all_ready(reports);folder=p.OUT/'dev_evaluation';folder.mkdir(parents=True,exist_ok=True);(folder/'figures').mkdir(exist_ok=True)
    for n in p.NAMES:
        if p.base.parent.digest(reports[n]['checkpoint'])!=reports[n]['checkpoint_sha256']:raise ValueError('FAIL_CHECKPOINT_CHANGED')
        m,c=training.load_final(reports[n]['checkpoint'],n)
        if c['bindings']!=training.bindings(a):raise ValueError('FAIL_FINAL_BINDING')
        del m,c;gc.collect();torch.cuda.empty_cache()
    guard.ready=True;dev=a['groups']['validation_normal']+a['groups']['excluded_train_defect']
    report=dict(**p.FLAGS,status='RUNNING',training=reports,metrics={},exact_metadata={},state_unchanged={})
    predictions=[];all_images=[];all_regions=[];bounds=a['parent']['size_groups']['boundaries']
    for n in p.NAMES:
        m,c=training.load_final(reports[n]['checkpoint'],n);m.requires_grad_(False);before=training.old.state_hash(m)
        hw=tuple(p.config(n)['input_hw']);native_dir=folder/'raw_native'/n.lower();native_dir.mkdir(parents=True,exist_ok=True);rows=[]
        for i,r in enumerate(dev):
            x=training.old.normal_input(m,r['image_path'],guard,'dev');o=forward(m,x,n,hw)
            shape=(int(r['height']),int(r['width']));pixel=p.base.native(o['pixel_map'],shape);image=p.base.native(o['image_map'],shape)
            path=native_dir/(r['id'].replace('/','__')+'.npy')
            if path.exists():
                if not np.array_equal(np.load(path,allow_pickle=False),pixel):raise ValueError('FAIL_EXISTING_DEV_MAP_CHANGED')
            else:np.save(path,pixel,allow_pickle=False)
            guard.check(r['mask_path'],'dev')
            rows.append(dict(source_id=r['id'],sample_id=r['id'],method=n,method_description=p.config(n)['method_description'],
                image_path=r['image_path'],gt_label=int(r['label']),gt_mask_path=r['mask_path'],native_shape=shape,
                raw_map_path=str(path),image_score=float(image.max()),size_group='ABCDE'[np.searchsorted(bounds,float(r['defect_ratio']),side='left')] if int(r['label']) else 'NORMAL'))
            del x,o,pixel,image
            if (i+1)%50==0:print(f'{n} 70K DEVELOPMENT {i+1}/455; official test forwards=0',flush=True)
        p.base.prediction_gate(a,rows)
        if training.old.state_hash(m)!=before:raise ValueError('FAIL_INFERENCE_STATE_CHANGED')
        report['state_unchanged'][n]=True;del m,c;gc.collect();torch.cuda.empty_cache()
        p.base.csv_save(folder/(n.lower()+'_dev_predictions.csv'),rows)
        v,images,regions,meta=metrics(rows,n,folder/'metric_workspace'/n.lower())
        if meta['connected_regions']!=275 or len(meta['regions'])!=455:raise ValueError('FAIL_DEV_COUNTS')
        report['metrics'][n]=v;report['exact_metadata'][n]=meta;predictions+=rows;all_images+=images;all_regions+=regions
        p.base.save(folder/(n.lower()+'_dev_metrics.json'),dict(**p.FLAGS,method_description=p.config(n)['method_description'],metrics=v,exact_metadata=meta))
    for filename,rows in [('dev_predictions.csv',predictions),('dev_image_metrics.csv',all_images),('dev_region_metrics.csv',all_regions)]:p.base.csv_save(folder/filename,rows)
    groups=[]
    for n in p.NAMES:
        for g in 'ABCDE':
            v=[r['PRO005'] for r in all_images if r['method']==n and r['size_group']==g]
            groups.append(dict(method=n,size_group=g,n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),std=float(np.std(v))))
    report['size_groups_description_only']=groups;p.base.csv_save(folder/'dev_size_group_analysis.csv',groups)
    data={n:{r['source_id']:r['PRO005'] for r in all_images if r['method']==n} for n in p.NAMES}
    ids=sorted(data['STD'])
    if set(ids)!=set(data['RECT']):raise ValueError('FAIL_PAIRED_IDS')
    report['paired']=paired([data['RECT'][i] for i in ids],[data['STD'][i] for i in ids])
    data={n:{(r['source_id'],r['region_id']):r['PRO005'] for r in all_regions if r['method']==n} for n in p.NAMES}
    s,r=data['STD'],data['RECT']
    if set(s)!=set(r):raise ValueError('FAIL_REGION_IDS')
    report['region_paired']=dict(RECT_ONLY=sum(r[k]>0 and s[k]<=0 for k in s),STD_ONLY=sum(r[k]<=0 and s[k]>0 for k in s),BOTH=sum(r[k]>0 and s[k]>0 for k in s),BOTH_BAD=sum(r[k]<=0 and s[k]<=0 for k in s))
    p.base.csv_save(folder/'comparison_table.csv',[dict(method=n,method_description=p.config(n)['method_description'],**report['metrics'][n]) for n in p.NAMES])
    report.update(access=guard.report(),**contamination_scan(a,folder))
    if guard.blocked or p.base.parent.digest(p.base.parent.OUT/'external_protocol.json')!=p.base.PARENT_SHA:raise ValueError('FAIL_INTEGRITY')
    if any(p.base.parent.digest(t['checkpoint'])!=t['checkpoint_sha256'] for t in reports.values()):raise ValueError('FAIL_CHECKPOINT_CHANGED')
    report.update(decision=p.decide(report['metrics'],a['pilot']['metrics']),status='PASS')
    p.base.save(folder/'dev_report.json',report);text=final_text(report);(folder/'dev_report.txt').write_text(text,encoding='utf-8');print(text,flush=True)
    return report
