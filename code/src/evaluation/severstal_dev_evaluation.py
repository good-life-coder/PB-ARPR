"""Development-only evaluation using the existing exact rank metric engine."""
import gc
import json
import time
from pathlib import Path
import numpy as np
import torch
from src.evaluation import severstal_protocol as p
from src.training import severstal_pilot as t
from src.evaluation.ksdd2_dev_evaluation import forward,paired

def contamination(a,rows):
    allowed={r['id'] for role in ('validation_normal','dev_anomaly') for r in a['groups'][role]}
    if any(r['source_id'] not in allowed for r in rows):raise ValueError('FAIL_BLIND_CONTAMINATION')
    return True

@torch.no_grad()
def runtime(a,guard):
    result={};r=a['groups']['validation_normal'][0]
    for name in ('STD','RECT'):
        m,c=t.load_final(name,a);before=t.old.state_hash(m);x=t.input_image(m,r,guard,'runtime');hw=tuple(a['methods'][name]['input_hw']);times=[]
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        for i in range(120):
            torch.cuda.synchronize();start=time.perf_counter();o=forward(m,x,name,hw);torch.cuda.synchronize()
            if i>=20:times.append((time.perf_counter()-start)*1000)
            del o
        if before!=t.old.state_hash(m):raise ValueError('FAIL_RUNTIME_STATE')
        result[name]=dict(mean_ms=float(np.mean(times)),median_ms=float(np.median(times)),FPS=1000/float(np.mean(times)),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,warmup=20,runs=100,cuda_synchronize=True,source_role='validation_normal')
        del m,c,x;gc.collect();torch.cuda.empty_cache()
    p.save(p.PILOT/'runtime_report.json',result);return result

@torch.no_grad()
def run(a,guard,reports):
    t.all_ready(reports)
    for name in ('STD','RECT'):
        m,c=t.load_final(name,a);del m,c;gc.collect();torch.cuda.empty_cache()
    out=p.PILOT/'dev_evaluation';out.mkdir(exist_ok=True)
    if (out/'first_dev_access.json').exists():raise ValueError('FAIL_DEV_ALREADY_STARTED')
    from datetime import datetime,timezone
    p.save(out/'first_dev_access.json',dict(time=datetime.now(timezone.utc).isoformat(),both_checkpoints_passed=True))
    guard.ready=True;source=a['groups']['validation_normal']+a['groups']['dev_anomaly'];data={};images=[];regions=[];predictions=[];states={}
    for row in source:guard.check(row['mask_path'],'dev');p.verify_mask(row)
    for name in ('STD','RECT'):
        m,c=t.load_final(name,a);before=t.old.state_hash(m);hw=tuple(a['methods'][name]['input_hw']);dest=out/'raw_native'/name.lower();dest.mkdir(parents=True,exist_ok=True);rows=[]
        for i,r in enumerate(source):
            x=t.input_image(m,r,guard,'dev');o=forward(m,x,name,hw);shape=(int(r['height']),int(r['width']));pixel=p.native(o['pixel_map'],shape);path=dest/(Path(r['id']).stem+'.npy')
            if pixel.dtype!=np.float32 or pixel.shape!=shape or not np.isfinite(pixel).all():raise ValueError('FAIL_NATIVE_MAP')
            with path.open('xb') as f:np.save(f,pixel,allow_pickle=False)
            guard.check(r['mask_path'],'dev')
            rows.append(dict(source_id=r['id'],sample_id=r['id'],method=name,image_path=r['image_path'],gt_label=int(r['label']),gt_mask_path=r['mask_path'],native_shape=shape,raw_map_path=str(path),image_score=float(pixel.max()),size_group='ABCDE'[np.searchsorted(a['protocol']['size_boundaries'],float(r['defect_ratio']),side='left')] if int(r['label']) else 'NORMAL'))
            del x,o,pixel
            if (i+1)%100==0:print(f'DEV {name} {i+1}/{len(source)}',flush=True)
        if before!=t.old.state_hash(m):raise ValueError('FAIL_STATE_CHANGED')
        states[name]=dict(before=before,after=before,unchanged=True);del m,c;gc.collect();torch.cuda.empty_cache()
        contamination(a,rows);p.csv_save(out/(name.lower()+'_predictions.csv'),rows)
        values,im,reg,meta=p.exact_metrics(rows,name,out/'metric_workspace'/name.lower());data[name]=values;images+=im;regions+=reg;predictions+=rows
        p.save(out/(name.lower()+'_metrics.json'),dict(metrics=values,metadata=meta))
    contamination(a,predictions);p.csv_save(out/'dev_predictions.csv',predictions);p.csv_save(out/'dev_image_metrics.csv',images);p.csv_save(out/'dev_region_metrics.csv',regions)
    groups=[]
    for name in ('STD','RECT'):
        for g in 'ABCDE':
            v=[r['PRO005'] for r in images if r['method']==name and r['size_group']==g]
            groups.append(dict(method=name,size_group=g,n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),std=float(np.std(v))))
    p.csv_save(out/'dev_size_group_analysis.csv',groups)
    by={n:{r['source_id']:r['PRO005'] for r in images if r['method']==n} for n in ('STD','RECT')};ids=sorted(by['STD'])
    pairs=paired([by['RECT'][i] for i in ids],[by['STD'][i] for i in ids])
    by={n:{(r['source_id'],r['region_id']):r['PRO005'] for r in regions if r['method']==n} for n in ('STD','RECT')};s,r=by['STD'],by['RECT']
    if set(s)!=set(r):raise ValueError('FAIL_REGION_PAIRING')
    rp=dict(RECT_ONLY=sum(r[k]>0 and s[k]<=0 for k in s),STD_ONLY=sum(r[k]<=0 and s[k]>0 for k in s),BOTH=sum(r[k]>0 and s[k]>0 for k in s),BOTH_BAD=sum(r[k]<=0 and s[k]<=0 for k in s))
    decision=p.decide(data['STD'],data['RECT']);timing=runtime(a,guard)
    if guard.blocked:raise ValueError('FAIL_BLIND_ACCESS_ATTEMPT')
    report=dict(status='PASS',**p.FLAGS,Approximation=False,training=reports,metrics=data,decision=decision,paired=pairs,region_paired=rp,size_groups=groups,runtime=timing,state_integrity=states,split_sha=a['protocol']['canonical_split_sha256'],geometry_sha=a['protocol']['geometry_sha256'],access=dict(allowed_reads=dict(guard.counts),blocked=guard.blocked))
    p.save(out/'dev_report.json',report);text=json.dumps(report,indent=2);(out/'dev_report.txt').write_text(text,encoding='utf-8');print(text,flush=True)
    if decision['RECT_PROMISING']:p.save(p.PILOT/'FINAL_SEVERSTAL_BLIND_PROTOCOL_DRAFT.json',dict(status='DRAFT',split_sha=report['split_sha'],geometry_sha=report['geometry_sha'],STD_input=[256,256],RECT_input=a['geometry']['rect_hw'],future_training_steps=70000,future_seeds=[42,43,44,45,46],metrics=['Image AUROC','Pixel AUROC','AUPRO005','AUPRO030'],blind_tuning_allowed=False,post_blind_modification_allowed=False))
    return report
