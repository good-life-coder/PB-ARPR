"""Development-only adapters around the existing native exact rank engine."""
from src.evaluation import exact_aupro as exact
import csv
import gc
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from src.evaluation import ksdd2_pilot_protocol as p
from src.training import ksdd2_pilot_training as training
from src.models import multiscale_st_s1 as s1
from src.evaluation import rectangular_formal_evaluation as reference

KEYS=('image_auroc','pixel_auroc','aupro005','aupro030')

def fuse_canonical(shallow,final,ae):
    # Frozen Phase15A Consensus formula, no competing heads or fusion selection.
    return dict(image_map=.5*final+.5*ae,pixel_map=p.parent.consensus(shallow,final,ae))

@torch.no_grad()
def forward(m,x,name,hw):
    if m.training or m.model.training:raise ValueError('FAIL_INFERENCE_MODE')
    with s1.Capture(m,s1.WANTED if name=='DGNB' else {}) as cap:
        student,distance=m.model.compute_student_teacher_distance(x)
    final,ae=m.model.compute_maps(x,student,distance,normalize=True)
    if name!='DGNB':return dict(image_map=.5*final+.5*ae,pixel_map=.5*final+.5*ae)
    raw=s1.s1_map(s1.discrepancy(cap.values['teacher',s1.S1],cap.values['student',s1.S1]))
    shallow=s1.normalized_map(raw,m.qa_s1,m.qb_s1)
    shallow=torch.nn.functional.interpolate(shallow,size=hw,mode='bilinear',align_corners=False)
    return fuse_canonical(shallow,final,ae)

def metrics(rows,name,folder):
    folder=Path(folder);meta={}
    exact.native(rows,meta,torch,cv2,np,Image,temp_dir=folder)
    bg=np.memmap(folder/'background_scores.dat',mode='r',dtype=np.float32,shape=(meta['background_pixels'],))
    pos=np.memmap(folder/'positive_scores.dat',mode='r',dtype=np.float32,shape=(meta['positive_pixels'],))
    percentiles=reference.diagnostic.percentiles
    auc=sum(float(percentiles(bg,pos[i:i+100000])[2].sum()) for i in range(0,len(pos),100000))/len(pos)/100
    neg=np.sort([r['image_score'] for r in rows if not int(r['gt_label'])]);positive=np.asarray([r['image_score'] for r in rows if int(r['gt_label'])])
    result=dict(image_auroc=float(percentiles(neg,positive)[2].mean()/100),pixel_auroc=auc,
        aupro005=meta['aupro_0.05'],aupro030=meta['aupro_0.30'],approximation=False)
    images=[];regions=[]
    for row in rows:
        if not int(row['gt_label']):continue
        a=np.load(row['raw_map_path'],allow_pickle=False);labels,sizes=exact.components(exact.get_mask(row,Image,np),cv2,np)
        fg=labels>0;values=a[fg];weights=1/((len(sizes)-1)*sizes[labels[fg]].astype(np.float64))
        points=exact.pro_at_fprs(bg,values,weights,[.05,.30],np)
        localbg=np.sort(a[~fg]);areas,_=exact.integrate(localbg,values,weights,np)
        images.append(dict(source_id=row['source_id'],method=name,size_group=row['size_group'],PRO005=float(points[0]),
            PRO030=float(points[1]),per_image_AUPRO005=areas[.05],per_image_AUPRO030=areas[.30],connected_regions=len(sizes)-1))
        for rid in range(1,len(sizes)):
            scores=a[labels==rid];w=np.full(len(scores),1/len(scores),np.float64)
            pro=exact.pro_at_fprs(bg,scores,w,[.05,.30],np)
            regions.append(dict(source_id=row['source_id'],method=name,region_id=rid,pixels=int(sizes[rid]),PRO005=float(pro[0]),PRO030=float(pro[1])))
    if any(not np.isfinite(result[k]) or not 0<=result[k]<=1 for k in KEYS):raise ValueError('FAIL_METRIC_RANGE')
    del bg,pos;gc.collect()
    return result,images,regions,meta

def paired(new,old):
    d=np.asarray(new)-np.asarray(old)
    result=dict(improved=int((d>0).sum()),worsened=int((d<0).sum()),unchanged=int((d==0).sum()),
        mean_delta_pp=float(d.mean()*100),median_delta_pp=float(np.median(d)*100))
    try:
        from scipy.stats import wilcoxon
        w=wilcoxon(d,zero_method='wilcox') if np.any(d) else None
        result['wilcoxon']=dict(statistic=float(w.statistic) if w else 0.,pvalue=float(w.pvalue) if w else 1.)
    except ImportError:result['wilcoxon']=None
    return result

def contamination_scan(a,folder):
    test=a['groups']['test'];tokens={r['id'] for r in test}|{str(Path(r['image_path'])) for r in test}
    stems={Path(r['image_path']).stem for r in test}
    for f in Path(folder).rglob('*'):
        if f.name in ('official_test','test_predictions','test_heatmaps'):raise ValueError('FAIL_TEST_CONTAMINATION')
        if f.suffix=='.npy' and f.stem.removeprefix('train__') in stems:raise ValueError('FAIL_TEST_CONTAMINATION')
        if f.suffix=='.csv':
            with f.open(encoding='utf-8') as stream:
                for row in csv.DictReader(stream):
                    if any(v in tokens or v in stems for k,v in row.items() if k in ('source_id','sample_id','source_image_path','image_path','source_filename')):
                        raise ValueError('FAIL_TEST_CONTAMINATION')
    return dict(OFFICIAL_TEST_FORWARD_COUNT=0,OFFICIAL_TEST_PREDICTIONS_CREATED=0,FINAL_KSDD2_TEST_UNTOUCHED=True)

def final_text(report):
    lines=['='*40,'PHASE 15B - KSDD2 10K PILOT','='*40,'Pilot: DEVELOPMENT ONLY',
        'Official test untouched: True','Train normal: 1876','Validation normal: 209','Dev anomalies: 246','Official test: 1004']
    for name in p.NAMES:
        t=report['training'][name];lines.extend(['-'*40,name+' 10K','Completed: '+str(t['completed']),
            f'Training time: {t["training_hours"]:.6f} h; {t["seconds_per_step"]:.6f} sec/step',
            f'Peak VRAM allocated/reserved: {t["peak_allocated_mib"]:.3f}/{t["peak_reserved_mib"]:.3f} MiB',
            'Checkpoint: '+t['checkpoint'],'SHA256: '+t['checkpoint_sha256'],'Strict reload: PASS',
            'Quantiles: '+json.dumps(t['quantiles'])])
        if name=='DGNB':lines.extend(['Lambda: '+json.dumps(t['lambda_stats']),
            f'Boundary hits: {t["boundary_hit_count"]}/10000 ({t["boundary_hit_percent"]:.4f}%)',
            'Gradient probes: '+json.dumps({k:v['medians'] for k,v in t['gradient_probes'].items()})])
    lines+=['-'*40,'DEV PERFORMANCE (percent)','Metric                 STD          RECT         DGNB']
    for k in KEYS:lines.append(k.ljust(20)+' '.join(f'{100*report["metrics"][n][k]:12.6f}' for n in p.NAMES))
    lines+=['-'*40,'DECISION',json.dumps(report['decision'],indent=2),
        'Official test forward count: 0','Official test predictions: 0','FINAL_KSDD2_TEST_UNTOUCHED: True',
        'DEV_NORMAL_REUSED_FOR_QUANTILE: True','PILOT_CHECKPOINT_ALLOWED_FOR_FINAL: False','STATUS: PASS','='*40]
    return '\n'.join(lines)

def run(a,guard,reports):
    p.all_ready(reports);folder=p.OUT/'dev_evaluation';folder.mkdir(parents=True,exist_ok=True);(folder/'figures').mkdir(exist_ok=True)
    # Recheck every frozen file and strict reload before first anomalous forward.
    for name in p.NAMES:
        path=Path(reports[name]['checkpoint'])
        if p.parent.digest(path)!=reports[name]['checkpoint_sha256']:raise ValueError('FAIL_FROZEN_CHECKPOINT_CHANGED')
        m,c=training.load_final(path,a['parent']['methods'][name]['input_hw'])
        p.checkpoint_binding(c,name,a['parent']['methods'][name]['input_hw'])
        del m,c;gc.collect();torch.cuda.empty_cache()
    guard.ready=True;dev=a['groups']['validation_normal']+a['groups']['excluded_train_defect']
    predictions=[];all_images=[];all_regions=[];report=dict(**p.FLAGS,status='RUNNING',training=reports,metrics={},exact_metadata={})
    bounds=a['parent']['size_groups']['boundaries']
    for name in p.NAMES:
        m,c=training.load_final(reports[name]['checkpoint'],a['parent']['methods'][name]['input_hw']);m.requires_grad_(False)
        before=training.state_hash(m);hw=tuple(a['parent']['methods'][name]['input_hw']);rows=[]
        native_dir=folder/'raw_native'/name.lower();native_dir.mkdir(parents=True,exist_ok=True)
        for i,r in enumerate(dev):
            x=training.normal_input(m,r['image_path'],guard,'dev');o=forward(m,x,name,hw)
            shape=(int(r['height']),int(r['width']));pixel=p.native(o['pixel_map'],shape);image=p.native(o['image_map'],shape)
            output=native_dir/(r['id'].replace('/','__')+'.npy')
            if output.exists():raise FileExistsError('FAIL_EXISTING_DEV_PREDICTION')
            np.save(output,pixel,allow_pickle=False)
            guard.check(r['mask_path'],'dev')
            row=dict(source_id=r['id'],sample_id=r['id'],method=name,image_path=r['image_path'],
                gt_label=int(r['label']),gt_mask_path=r['mask_path'],native_shape=shape,raw_map_path=str(output),
                image_score=float(image.max()),size_group='ABCDE'[np.searchsorted(bounds,float(r['defect_ratio']),side='left')] if int(r['label']) else 'NORMAL')
            rows.append(row);del x,o,pixel,image
            if (i+1)%50==0:print(f'{name} DEVELOPMENT inference {i+1}/455; official test forwards=0',flush=True)
        p.prediction_gate(a,rows)
        if training.state_hash(m)!=before:raise ValueError('FAIL_DEV_STATE_MUTATION')
        del m,c;gc.collect();torch.cuda.empty_cache()
        p.csv_save(folder/(name.lower()+'_dev_predictions.csv'),rows)
        values,images,regions,meta=metrics(rows,name,folder/'metric_workspace'/name.lower())
        report['metrics'][name]=values;report['exact_metadata'][name]=meta
        predictions.extend(rows);all_images.extend(images);all_regions.extend(regions)
        p.save(folder/(name.lower()+'_dev_metrics.json'),dict(**p.FLAGS,metrics=values,exact_metadata=meta))
    p.csv_save(folder/'dev_predictions.csv',predictions);p.csv_save(folder/'dev_image_metrics.csv',all_images);p.csv_save(folder/'dev_region_metrics.csv',all_regions)
    groups=[]
    for name in p.NAMES:
        for group in 'ABCDE':
            v=[r['PRO005'] for r in all_images if r['method']==name and r['size_group']==group]
            groups.append(dict(method=name,size_group=group,n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),std=float(np.std(v))))
    p.csv_save(folder/'dev_size_group_analysis.csv',groups);report['size_groups_description_only']=groups
    report['paired']={};report['region_paired']={}
    for name,base in [('RECT','STD'),('DGNB','RECT'),('DGNB','STD')]:
        new={r['source_id']:r['PRO005'] for r in all_images if r['method']==name};old={r['source_id']:r['PRO005'] for r in all_images if r['method']==base}
        report['paired'][name+'_vs_'+base]=paired([new[k] for k in sorted(new)],[old[k] for k in sorted(new)])
        nr={(r['source_id'],r['region_id']):r['PRO005'] for r in all_regions if r['method']==name};br={(r['source_id'],r['region_id']):r['PRO005'] for r in all_regions if r['method']==base}
        if set(nr)!=set(br):raise ValueError('FAIL_REGION_PAIRING')
        report['region_paired'][name+'_vs_'+base]=dict(NEW_ONLY=sum(nr[k]>0 and br[k]<=0 for k in nr),REFERENCE_ONLY=sum(nr[k]<=0 and br[k]>0 for k in nr),BOTH=sum(nr[k]>0 and br[k]>0 for k in nr),BOTH_BAD=sum(nr[k]<=0 and br[k]<=0 for k in nr))
    p.csv_save(folder/'comparison_table.csv',[dict(method=n,**report['metrics'][n]) for n in p.NAMES])
    report.update(decision=p.decide(report['metrics']),access=guard.report(),**contamination_scan(a,folder),status='PASS')
    if guard.blocked or p.parent.digest(p.parent.OUT/'external_protocol.json')!=p.PARENT_SHA:raise ValueError('FAIL_INTEGRITY')
    p.save(folder/'dev_report.json',report);text=final_text(report);(folder/'dev_report.txt').write_text(text,encoding='utf-8');print(text,flush=True)
    return report
