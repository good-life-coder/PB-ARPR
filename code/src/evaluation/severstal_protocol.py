"""Severstal official annotated-training audit and immutable one-class roles."""
import csv
import hashlib
import json
import math
import sys
from collections import Counter,defaultdict
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from src.common.pbarpr_geometry import select_rect,feature_shapes
from src.evaluation.ksdd2_external_protocol import ROOT,digest
from src.evaluation.ksdd2_pilot_protocol import save,csv_save,read,native
from src.evaluation.ksdd2_dev_evaluation import metrics as exact_metrics

OUT=ROOT/'experiments/phase16a_severstal_audit'
PILOT=ROOT/'experiments/phase16b_severstal_10k_pilot'
DATA=ROOT/'datasets/Severstal'
ROLES=('train_normal','validation_normal','dev_anomaly','final_blind_normal','final_blind_anomaly')
FLAGS=dict(KAGGLE_COMPETITION_TEST_USED=False,FINAL_BLIND_FORWARD_COUNT=0,FINAL_BLIND_PREDICTIONS_CREATED=0,
 SEVERSTAL_FINAL_BLIND_UNTOUCHED=True,DEV_NORMAL_REUSED_FOR_QUANTILE=True,PILOT_FOR_METHOD_DEVELOPMENT=True,
 DO_NOT_REPORT_AS_FINAL_SEVERSTAL_RESULT=True,DGNB_ENABLED=False)

def schema(columns):
    if set(columns)=={'ImageId','ClassId','EncodedPixels'}:return 'separate'
    if set(columns)=={'ImageId_ClassId','EncodedPixels'}:return 'combined'
    raise ValueError('FAIL_CSV_SCHEMA')

def decode(rle,h,w):
    a=np.zeros(h*w,np.uint8)
    if not rle or not str(rle).strip():return a.reshape((h,w),order='F')
    try:v=[int(x) for x in str(rle).split()]
    except ValueError:raise ValueError('FAIL_RLE_INTEGER')
    if len(v)%2:raise ValueError('FAIL_RLE_PAIRS')
    end=0
    for start,length in zip(v[::2],v[1::2]):
        if start<1 or length<=0 or start+length-1>h*w or start-1<end:raise ValueError('FAIL_RLE_BOUNDS_OR_OVERLAP')
        a[start-1:start-1+length]=1;end=start-1+length
    return a.reshape((h,w),order='F')

def union(rles,h,w):
    a=np.zeros((h,w),np.uint8)
    for rle in rles:a|=decode(rle,h,w)
    return a

def dedup(rows):
    groups=defaultdict(list)
    for r in rows:groups[r['image_sha']].append(r)
    kept=[];excluded=[]
    for group in groups.values():
        group=sorted(group,key=lambda r:r['id'])
        if len({r['mask_sha'] for r in group})!=1:raise ValueError('FAIL_CONFLICTING_DUPLICATE')
        kept.append(group[0]);excluded.extend(dict(id=r['id'],canonical_id=group[0]['id'],image_sha=r['image_sha'],mask_sha=r['mask_sha']) for r in group[1:])
    return sorted(kept,key=lambda r:r['id']),excluded

def make_split(rows):
    rows=sorted(rows,key=lambda r:r['id']);rng=np.random.default_rng(42);normal=[r for r in rows if not int(r['label'])]
    order=rng.permutation(len(normal));roles={}
    nt=math.floor(.7*len(normal));nv=math.floor(.1*len(normal))
    for i,index in enumerate(order):roles[normal[index]['id']]='train_normal' if i<nt else 'validation_normal' if i<nt+nv else 'final_blind_normal'
    defects=[r for r in rows if int(r['label'])];counts=Counter(r['signature'] for r in defects);strata=defaultdict(list)
    for r in defects:strata[r['signature'] if counts[r['signature']]>=10 else 'RARE_SIGNATURE'].append(r)
    alloc={k:math.floor(.2*len(v)) for k,v in strata.items()};remainder=math.floor(.2*len(defects))-sum(alloc.values())
    for k in sorted(strata,key=lambda k:(-(.2*len(strata[k])-alloc[k]),k))[:remainder]:alloc[k]+=1
    stratum={}
    for k in sorted(strata):
        group=strata[k]
        for i,index in enumerate(rng.permutation(len(group))):
            sid=group[index]['id'];roles[sid]='dev_anomaly' if i<alloc[k] else 'final_blind_anomaly';stratum[sid]=k
    return [dict(r,split=roles[r['id']],stratum=stratum.get(r['id'],'NORMAL')) for r in rows]

def split_hash(rows):
    return hashlib.sha256(json.dumps([{k:str(r[k]) for k in ('id','split','image_sha','mask_sha','signature')} for r in sorted(rows,key=lambda r:r['id'])],sort_keys=True).encode()).hexdigest()

def size_bins(rows):return np.quantile([float(r['defect_ratio']) for r in rows if r['split']=='dev_anomaly'],[.2,.4,.6,.8]).tolist()

def verify_mask(row):
    with Image.open(row['mask_path']) as im:a=np.asarray(im)
    if a.shape!=(int(row['height']),int(row['width'])) or not np.isin(a,[0,255]).all():raise ValueError('FAIL_MASK_INTEGRITY')
    binary=(a>0).astype(np.uint8)
    if hashlib.sha256(binary.tobytes()).hexdigest()!=row['mask_sha']:raise ValueError('FAIL_MASK_INTEGRITY')
    return True

def geometry_regression():
    a=read(ROOT/'experiments/phase15a_ksdd2_audit/dataset_audit.json')
    hw=select_rect(a['aspect_hw']['median'])
    if hw!=(864,320):raise ValueError('FAIL_GEOMETRY_REGRESSION')
    return hw

def config(hw):
    common=dict(parameters=8058628,seed=42,batch=1,precision='FP32',steps=10000,loss='unchanged official EfficientAD',scheduler_step=66500,early_stopping=False)
    return dict(STD=dict(common,input_hw=[256,256],user_proposed=False),RECT=dict(common,input_hw=list(hw),user_proposed=True))

def geometry_scales(target,median_h,median_w):
    sy=target[0]/median_h;sx=target[1]/median_w
    return dict(scale_x=sx,scale_y=sy,anisotropy=max(sx,sy)/min(sx,sy),pixels=math.prod(target),feature_shapes=feature_shapes(target),model_final_stride=4,model_final_RF=33,model_s1_stride=1,model_s1_RF=4,native_stride_y=4/sy,native_stride_x=4/sx,native_final_RF_y=33/sy,native_final_RF_x=33/sx,native_s1_RF_y=4/sy,native_s1_RF_x=4/sx)

def decide(s,r):
    d={k:100*(r[k]-s[k]) for k in ('image_auroc','pixel_auroc','aupro005','aupro030')};e=1e-12
    promising=(d['aupro005']>=1-e or (d['pixel_auroc']>=1-e and d['aupro005']>=-.5-e)) and d['image_auroc']>=-5-e
    strong=d['aupro005']>=3-e and d['pixel_auroc']>=1-e and d['image_auroc']>=-2-e
    negative=not promising and d['aupro005']<=-1+e and d['pixel_auroc']<=e
    return dict(delta_pp=d,RECT_PROMISING=promising,RECT_STRONG=strong,RECT_NEGATIVE=negative,RECT_INCONCLUSIVE=not(promising or negative),RECOMMEND_70K=promising,STOP_SEVERSTAL_RECT_LINE=negative,NEXT_ACTION='RECOMMEND_FRESH_70K' if promising else 'STOP_SEVERSTAL_RECT_LINE' if negative else 'CONSIDER_20K_PILOT')

class Guard:
    def __init__(self,rows):
        self.ready=False;self.counts=Counter();self.blocked=0
        self.by={str(Path(r[k]).resolve()).casefold():r for r in rows for k in ('image_path','mask_path')}
    def check(self,path,purpose):
        r=self.by.get(str(Path(path).resolve()).casefold())
        if r is None:raise ValueError('FAIL_UNREGISTERED_SOURCE')
        if 'blind' in r['split']:raise ValueError('FAIL_BLIND_ACCESS')
        allowed=dict(train=['train_normal'],probe=['train_normal'],quantile=['validation_normal'],dev=['validation_normal','dev_anomaly'],runtime=['validation_normal'])
        if r['split'] not in allowed.get(purpose,[]) or purpose=='dev' and not self.ready:raise ValueError('FAIL_SOURCE_ROLE')
        self.counts[purpose]+=1;return r['id']
    def install(self):
        forbidden={k for k,r in self.by.items() if 'blind' in r['split']}
        def hook(event,args):
            if event=='open' and isinstance(args[0],(str,bytes,Path)) and str(Path(args[0]).resolve()).casefold() in forbidden:
                self.blocked+=1;raise PermissionError('FAIL_BLIND_FILE_ACCESS')
        sys.addaudithook(hook)

def assets():
    rows=list(csv.DictReader((OUT/'split_manifest.csv').open(encoding='utf-8')));protocol=read(OUT/'split_protocol.json');geo=read(OUT/'geometry_protocol.json')
    if split_hash(rows)!=protocol['canonical_split_sha256'] or digest(OUT/'geometry_protocol.json')!=protocol['geometry_sha256']:raise ValueError('FAIL_FROZEN_ASSETS')
    if digest(OUT/'split_manifest.csv')!=protocol['manifest_file_sha256']:raise ValueError('FAIL_MANIFEST_BYTES')
    if dict(Counter(r['split'] for r in rows))!=protocol['counts'] or any(int(r['label'])!=int(r['split'] in ('dev_anomaly','final_blind_anomaly')) for r in rows):raise ValueError('FAIL_ROLE_LABEL_COUNTS')
    if read(OUT/'dataset_audit.json')['status']!='PASS':raise ValueError('FAIL_DATA_AUDIT')
    return dict(rows=rows,groups={k:[r for r in rows if r['split']==k] for k in ROLES},protocol=protocol,geometry=geo,methods=config(geo['rect_hw']))

def audit():
    if (OUT/'split_protocol.json').exists():return assets()
    OUT.mkdir(parents=True,exist_ok=True);source=DATA/'train.csv';images=sorted((DATA/'train_images').glob('*.jpg'))
    raw=list(csv.DictReader(source.open(encoding='utf-8-sig')));columns=list(raw[0]);kind=schema(columns);annotations=defaultdict(dict)
    for r in raw:
        sid,cls=(r['ImageId'],r['ClassId']) if kind=='separate' else r['ImageId_ClassId'].rsplit('_',1)
        cls=int(cls)
        if cls not in (1,2,3,4) or cls in annotations[sid]:raise ValueError('FAIL_DUPLICATE_OR_INVALID_CLASS_ANNOTATION')
        annotations[sid][cls]=r['EncodedPixels'].strip()
    orphan=set(annotations)-{f.name for f in images}
    if orphan:raise ValueError('FAIL_ORPHAN_ANNOTATION')
    discovery=dict(csv_columns=columns,csv_rows=len(raw),unique_annotation_image_ids=len(annotations),image_count=len(images),csv_bytes=source.stat().st_size,csv_sha256=digest(source),annotation_representation='sparse positive-only' if all(r['EncodedPixels'].strip() for r in raw) else 'explicit empty records',**FLAGS)
    save(OUT/'dataset_discovery_available.json',discovery);print(json.dumps(discovery,indent=2),flush=True)
    maskdir=OUT/'union_masks';maskdir.mkdir(exist_ok=True);rows=[];ann=[]
    for i,f in enumerate(images):
        with Image.open(f) as im:im.load();w,h=im.size
        ac=annotations.get(f.name,{});m=union(ac.values(),h,w)
        if m.shape!=(h,w) or not np.isin(m,[0,1]).all():raise ValueError('FAIL_MASK')
        active=[c for c,rle in ac.items() if decode(rle,h,w).any()];signature=''.join(str(int(c in active)) for c in (1,2,3,4))
        maskpath=maskdir/(f.stem+'.png');Image.fromarray(m*255).save(maskpath)
        regions=cv2.connectedComponents(m,connectivity=8)[0]-1;pixels=int(m.sum())
        rows.append(dict(id=f.name,image_path=str(f.resolve()),mask_path=str(maskpath.resolve()),image_sha=digest(f),mask_sha=hashlib.sha256(m.tobytes()).hexdigest(),file_bytes=f.stat().st_size,height=h,width=w,aspect_hw=h/w,label=int(pixels>0),signature=signature,defect_pixels=pixels,defect_ratio=pixels/(h*w),regions=regions,annotation_record=f.name in annotations))
        ann.append(dict(id=f.name,classes=';'.join(map(str,sorted(active))),signature=signature,rle_valid=True,pixels=pixels,mask_sha=rows[-1]['mask_sha']))
        if (i+1)%1000==0:print(f'AUDIT {i+1}/{len(images)}',flush=True)
    effective,duplicates=dedup(rows);csv_save(OUT/'image_manifest.csv',rows);csv_save(OUT/'annotation_manifest.csv',ann)
    if duplicates:csv_save(OUT/'duplicate_manifest.csv',duplicates)
    else:(OUT/'duplicate_manifest.csv').write_text('id,canonical_id,image_sha,mask_sha\n',encoding='utf-8')
    ratios=[r['defect_ratio'] for r in effective if r['label']];aspect=float(np.median([r['aspect_hw'] for r in effective]));median_h=float(np.median([r['height'] for r in effective]));median_w=float(np.median([r['width'] for r in effective]))
    report=dict(status='PASS',**discovery,effective_images=len(effective),normal=sum(not r['label'] for r in effective),defect=sum(r['label'] for r in effective),class_counts={str(c):sum(r['signature'][c-1]=='1' for r in effective) for c in range(1,5)},multi_class=sum(r['signature'].count('1')>1 for r in effective),duplicate_excluded=len(duplicates),native_sizes=dict(Counter(f"{r['height']}x{r['width']}" for r in effective)),median_h=median_h,median_w=median_w,median_aspect_hw=aspect,min_aspect_hw=min(r['aspect_hw'] for r in effective),max_aspect_hw=max(r['aspect_hw'] for r in effective),connected_regions=sum(r['regions'] for r in effective),defect_ratio=dict(min=min(ratios),median=float(np.median(ratios)),mean=float(np.mean(ratios)),max=max(ratios)),image_without_csv_record=sum(not r['annotation_record'] for r in rows),missing_record_interpretation='normal under sparse official annotation',corrupted_images=0,orphan_annotations=0,mask_mismatch=0,RLE_PASS=True)
    save(OUT/'dataset_audit.json',report);(OUT/'dataset_audit.txt').write_text(json.dumps(report,indent=2),encoding='utf-8')
    geometry_regression();hw=select_rect(aspect)
    def scales(target):
        return geometry_scales(target,median_h,median_w)
    geo=dict(FROZEN=True,rect_hw=list(hw),native_median_hw=[median_h,median_w],median_aspect_hw=aspect,STD=scales((256,256)),RECT=scales(hw),pixel_budget_deviation=(math.prod(hw)-262144)/262144,geometry_rule_reused=True,KSDD2_regression=[864,320],source_function='src.evaluation.ksdd2_external_protocol.select_rect',source_file_sha=digest(ROOT/'src/evaluation/ksdd2_external_protocol.py'),rule=read(ROOT/'experiments/phase15a_ksdd2_audit/external_protocol.json')['rect_algorithm'])
    save(OUT/'geometry_protocol.json',geo);split=make_split(effective);csv_save(OUT/'split_manifest.csv',split)
    protocol=dict(FROZEN=True,split_seed=42,canonical_split_sha256=split_hash(split),counts=dict(Counter(r['split'] for r in split)),normal_rule='floor70% train/floor10% validation/remainder blind',defect_rule='floor20% total dev, largest-remainder allocation by signature stratum; count<10 pooled; seed42 shuffle',size_boundaries=size_bins(split),size_boundaries_source='DEV_ANOMALY only',size_groups_used_for_decision=False,geometry_sha256=digest(OUT/'geometry_protocol.json'),csv_sha256=discovery['csv_sha256'],**FLAGS)
    protocol['manifest_file_sha256']=digest(OUT/'split_manifest.csv')
    save(OUT/'split_protocol.json',protocol);print(json.dumps(dict(split=protocol,geometry=geo),indent=2),flush=True);return assets()
