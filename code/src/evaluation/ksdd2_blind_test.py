"""One-shot frozen official inference; all gates precede model access to test."""
import csv
import gc
import json
import shutil
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from src.evaluation import ksdd2_repeat_summary as repeat
from src.training import ksdd2_repeat_training as loader
from src.evaluation.ksdd2_dev_evaluation import forward,metrics,paired
from src.evaluation import ksdd2_paper_summary as paper

base=repeat.base
OUT=base.ROOT/'experiments/phase15e_ksdd2_blind_test'
PROTOCOL=repeat.OUT/'summary/FINAL_BLIND_TEST_PROTOCOL.json'
native=base.native
EXPECTED_SHA=dict(zip([f'{s}_{n}' for s in repeat.SEEDS for n in repeat.NAMES],[
'62c708d75426b9352bb5cc434678faff2a90ea9203b987ffb06e28c37dbe0bec','b4428814ddbf83fde150353b1477ac9342e30962b7e56c6a1f12cf20bf371c41',
'0aeec7a7e833f39c8ad4f35ed9da2ced39be2a504d0c642b51248a1ca53a00ba','b84adee1bd935d34d2f87815c0a4335c738e2dd3452f55f2ccdfba39a4f31b7d',
'2e92f7faf19c487f3a34e6328e9d60cb08f6dd744a68ea7ead9a929d69eb5e8a','e039065ba4558476a833eac75a8e3e2dde00d35fe993769c5c32479a7d309fe7',
'ce6db5d44a450ab20c3a2eb05ca6d335b0184ffbfb0cbf05ca79312ae6b66eb5','f99def128021e75035df3c472f324aef2a39eb0a21e07c2ac614bfe185f1daf7',
'bb24cb277407c61fd6bfdcafc761dd70f0d80af1cf061167412ad55a0f907a0d','9efead350ed60465ffae86d23fe851d76d50774f5cadbc58c82d9628aacc38a7']))
RUNTIME=dict(seed=42,warmup=20,model_rounds=3,e2e_rounds=2,sources=1004,cuda_synchronize=True)
FLAGS=dict(THIS_IS_FIRST_OFFICIAL_TEST_FORWARD=True,OFFICIAL_TEST_USED_FOR_METHOD_SELECTION=False,
    METHOD_FROZEN_BEFORE_TEST=True,NO_POST_TEST_TUNING=True,POST_TEST_TRAINING_STARTED=False,
    FINAL_PAPER_METHOD_NAME_FROZEN=False,Approximation=False,DGNB_ENABLED=False)
FILES=['src/evaluation/ksdd2_blind_test.py','src/evaluation/ksdd2_paper_summary.py','scripts/evaluate_ksdd2_blind_test.py',
    'src/evaluation/ksdd2_dev_evaluation.py','src/evaluation/exact_aupro.py','src/evaluation/ksdd2_pilot_protocol.py']

def now():return datetime.now(timezone.utc).isoformat()
def geometry(name):
    if name not in repeat.NAMES:raise ValueError('FAIL_METHOD')
    return (256,256) if name=='STD' else (864,320)

def validate_protocol(p):
    if p!=repeat.blind_protocol():raise ValueError('FAIL_FROZEN_PROTOCOL')
    return dict(FROZEN=True,parent_has_explicit_frozen_field='FROZEN' in p,protocol=p)

def verify_sha(path,expected):
    if not Path(path).is_file() or base.parent.digest(path)!=expected:raise ValueError('FAIL_CHECKPOINT_GATE '+str(path))
    return True

def test_rows(a):
    rows=a['groups']['test']
    if a['split_sha']!=base.SPLIT_SHA or len(rows)!=1004 or len({r['id'] for r in rows})!=1004 or sum(int(r['label']) for r in rows)!=110:
        raise ValueError('FAIL_TEST_SPLIT')
    if any(r['official_split']!='test' or r['split']!='test' for r in rows):raise ValueError('FAIL_TEST_ROLE')
    return rows

def size_group(label,ratio,bounds):return 'ABCDE'[np.searchsorted(bounds,ratio,side='left')] if int(label) else 'NORMAL'

def map_gate(a,hw):
    if a.dtype!=np.float32 or a.shape!=tuple(hw) or not np.isfinite(a).all():raise ValueError('FAIL_NATIVE_MAP')
    return True

def checkpoint_path(seed,name):
    return (repeat.parent.OUT/repeat.parent.FOLDERS[name] if seed==42 else repeat.folder(seed,name))/'final.ckpt'

def load(seed,name):
    verify_sha(checkpoint_path(seed,name),EXPECTED_SHA[f'{seed}_{name}'])
    m,c=loader.core.load_final(checkpoint_path(seed,name),name) if seed==42 else loader.load_final(seed,name)
    m.eval();m.requires_grad_(False)
    if sum(v.numel() for v in m.parameters())!=8058628 or any(v.dtype!=torch.float32 for v in m.parameters()):raise ValueError('FAIL_CHECKPOINT_GATE')
    if c['global_step']!=70000 or c['quantiles']!=loader.old.quantile_values(m,name):raise ValueError('FAIL_CHECKPOINT_GATE')
    return m,c

class TestGuard:
    def __init__(self,rows):
        self.ready=False;self.paths={str(Path(r[k]).resolve()).casefold():r['id'] for r in rows for k in ('image_path','mask_path')}
    def check(self,path,purpose):
        key=str(Path(path).resolve()).casefold()
        if not self.ready or purpose!='official' or key not in self.paths:raise ValueError('FAIL_OFFICIAL_ACCESS')
        return self.paths[key]

class Journal:
    def __init__(self,folder,binding):
        self.folder=Path(folder);self.binding=binding;self.started=False;self.seen=set();self.counts={'prediction':0,'runtime':0}
    def start(self):
        self.folder.mkdir(parents=True,exist_ok=True)
        if any((self.folder/'raw_native').rglob('*.npy')):raise FileExistsError('FAIL_EXISTING_PREDICTIONS')
        self.audit=dict(**FLAGS,OFFICIAL_TEST_FIRST_ACCESS_TIMESTAMP=now(),PRE_TEST_FORWARD_COUNT=0,PRE_TEST_PREDICTIONS=0,binding=self.binding,status='RUNNING')
        with (self.folder/'blind_test_audit.json').open('x',encoding='utf-8') as f:json.dump(self.audit,f,indent=2)
        self.started=True
    def forward(self,model,sid,purpose):
        if not self.started or purpose not in self.counts:raise ValueError('FAIL_ACCESS_BEFORE_GATE')
        key=(model,sid)
        if purpose=='prediction' and key in self.seen:raise ValueError('FAIL_DUPLICATE_FORWARD')
        if purpose=='prediction':self.seen.add(key)
        self.counts[purpose]+=1
        with (self.folder/'forward_journal.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=now(),model=model,source_id=sid,purpose=purpose))+'\n')
    def finish(self,status):
        self.audit.update(status=status,forward_counts=self.counts,OFFICIAL_TEST_FORWARD_COUNT=sum(self.counts.values()),
            OFFICIAL_TEST_PREDICTIONS_CREATED=len(list((self.folder/'raw_native').rglob('*.npy'))),finished=now())
        base.save(self.folder/'blind_test_audit.json',self.audit)

def disk_gate(rows,free_bytes):
    # Retain ten raw float32 maps and exact-rank workspaces, plus safety margin.
    pixels=sum(int(r['height'])*int(r['width']) for r in rows)
    positives=sum(int(r['defect_pixels']) for r in rows)
    required=int(1.2*10*(8*pixels+8*positives))+2*1024**3
    if free_bytes<required:raise ValueError('FAIL_DISK_CAPACITY')
    return required

def precheck():
    if (OUT/'blind_test_audit.json').exists():raise FileExistsError('FAIL_BLIND_TEST_ALREADY_STARTED; use saved results, never rerun')
    a=repeat.base.assets();rows=test_rows(a);attest=validate_protocol(base.read(PROTOCOL))
    free=shutil.disk_usage(base.ROOT).free;required=disk_gate(rows,free)
    prior=base.read(repeat.OUT/'completion_verification.json');dev=base.read(repeat.OUT/'summary/repeat_report.json')
    if prior['status']!='PASS' or not prior['contamination']['FINAL_KSDD2_TEST_UNTOUCHED'] or not dev['READY_FOR_5SEED_ONE_SHOT_TEST']:raise ValueError('FAIL_PRE_TEST_INTEGRITY')
    maskrows={r['id']:r for r in csv.DictReader((base.parent.OUT/'mask_manifest.csv').open(encoding='utf-8'))}
    content=[]
    for r in rows:
        mr=maskrows[r['id']]
        if Path(mr['path']).resolve()!=Path(r['mask_path']).resolve():raise ValueError('FAIL_MASK_PATH')
        if base.parent.digest(r['image_path'])!=r['image_sha'] or base.parent.digest(r['mask_path'])!=mr['sha']:raise ValueError('FAIL_TEST_CONTENT')
        content.append(dict(id=r['id'],image_sha=r['image_sha'],mask_sha=mr['sha']))
    checks={}
    for seed in repeat.SEEDS:
        for name in repeat.NAMES:
            m,c=load(seed,name);k=f'{seed}_{name}'
            checks[k]=dict(path=str(checkpoint_path(seed,name)),SHA=EXPECTED_SHA[k],strict_reload=True,quantiles=c['quantiles'],parameters=8058628,state_hash=loader.old.state_hash(m))
            del m,c;gc.collect();torch.cuda.empty_cache()
    binding=dict(protocol_sha=base.parent.digest(PROTOCOL),split_sha=a['split_sha'],checkpoints=checks,content=content,
        code={f:base.parent.digest(base.ROOT/f) for f in FILES},dev_report_sha=base.parent.digest(repeat.OUT/'summary/repeat_report.json'))
    base.save(OUT/'frozen_protocol_attestation.json',dict(**attest,parent_sha=binding['protocol_sha']))
    base.save(OUT/'precheck_report.json',dict(status='PASS',binding=binding,RUNTIME=RUNTIME,disk=dict(free_bytes=free,required_bytes=required),READY_FOR_FIRST_BLIND_FORWARD=True))
    print('10 checkpoints: PASS\nCanonical split: PASS\nOfficial test: 1004\nNormal: 894\nDefect: 110\nMETHOD_FROZEN: True\nTest quantile fitting: False\nTest tuning: False\nPOST_TEST_TRAINING_ALLOWED: False\nREADY_FOR_FIRST_BLIND_FORWARD: True',flush=True)
    return a,rows,binding

def time_stats(values):
    v=np.asarray(values)*1000
    return dict(mean_ms=float(v.mean()),median_ms=float(np.median(v)),std_ms=float(v.std()),p90_ms=float(np.percentile(v,90)),p95_ms=float(np.percentile(v,95)),FPS=float(1000/v.mean()),runs=len(v))

@torch.no_grad()
def benchmark(rows,guard,journal):
    result={}
    for name in repeat.NAMES:
        m,c=load(42,name);before=loader.old.state_hash(m);key=f'42_{name}';hw=geometry(name)
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        for r in rows[:20]:
            x=loader.old.normal_input(m,r['image_path'],guard,'official');journal.forward(key,r['id'],'runtime');torch.cuda.synchronize();forward(m,x,name,hw);torch.cuda.synchronize();del x
        times=[]
        for epoch in range(3):
            for r in rows:
                x=loader.old.normal_input(m,r['image_path'],guard,'official');journal.forward(key,r['id'],'runtime')
                torch.cuda.synchronize();start=time.perf_counter();o=forward(m,x,name,hw);torch.cuda.synchronize();times.append(time.perf_counter()-start);del o,x
            print(f'RUNTIME {name} model-only {epoch+1}/3',flush=True)
        endtimes=[]
        for epoch in range(2):
            for r in rows:
                journal.forward(key,r['id'],'runtime');torch.cuda.synchronize();start=time.perf_counter()
                x=loader.old.normal_input(m,r['image_path'],guard,'official');o=forward(m,x,name,hw);a=native(o['pixel_map'],(int(r['height']),int(r['width'])));score=float(a.max())
                torch.cuda.synchronize();endtimes.append(time.perf_counter()-start);del a,o,x,score
            print(f'RUNTIME {name} end-to-end {epoch+1}/2',flush=True)
        result[name]=dict(model_only=time_stats(times),end_to_end=time_stats(endtimes),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        if before!=loader.old.state_hash(m):raise ValueError('FAIL_RUNTIME_STATE')
        del m,c;gc.collect();torch.cuda.empty_cache()
    base.save(OUT/'runtime_report.json',dict(protocol=RUNTIME,results=result));return result

@torch.no_grad()
def run(a,rows,binding):
    journal=Journal(OUT,binding);guard=TestGuard(rows);journal.start();guard.ready=True
    try:
        data={};groups={};paired_results={};region_results={};all_predictions=[];all_images=[];all_regions=[];states={}
        bounds=a['parent']['size_groups']['boundaries']
        for seed in repeat.SEEDS:
            data[str(seed)]={};seedimages=[];seedregions=[]
            for name in repeat.NAMES:
                key=f'{seed}_{name}';m,c=load(seed,name);before=loader.old.state_hash(m);pred=[];dest=OUT/'raw_native'/f'seed{seed}'/name.lower();dest.mkdir(parents=True,exist_ok=True)
                for i,r in enumerate(rows):
                    x=loader.old.normal_input(m,r['image_path'],guard,'official');journal.forward(key,r['id'],'prediction');o=forward(m,x,name,geometry(name))
                    hw=(int(r['height']),int(r['width']));pixel=native(o['pixel_map'],hw);map_gate(pixel,hw);path=dest/(r['id'].replace('/','__')+'.npy')
                    with path.open('xb') as f:np.save(f,pixel,allow_pickle=False)
                    guard.check(r['mask_path'],'official')
                    pred.append(dict(seed=seed,method=name,source_id=r['id'],sample_id=r['id'],image_path=r['image_path'],gt_label=int(r['label']),gt_mask_path=r['mask_path'],native_shape=hw,raw_map_path=str(path),image_score=float(pixel.max()),size_group=size_group(r['label'],float(r['defect_ratio']),bounds),map_sha=base.parent.digest(path)))
                    del x,o,pixel
                    if (i+1)%100==0:print(f'OFFICIAL Seed{seed} {name} {i+1}/1004',flush=True)
                if before!=loader.old.state_hash(m):raise ValueError('FAIL_STATE_CHANGED')
                states[key]=dict(before=before,after=before,unchanged=True);del m,c;gc.collect();torch.cuda.empty_cache()
                base.csv_save(OUT/f'seed{seed}_{name.lower()}_predictions.csv',pred)
                values,im,reg,meta=metrics(pred,name,OUT/'metric_workspace'/key)
                if len(im)!=110:raise ValueError('FAIL_DEFECT_COUNT')
                data[str(seed)][name]=values;seedimages+=im;seedregions+=reg;all_predictions+=pred
                base.save(OUT/f'{key}_metrics.json',dict(metrics=values,connected_regions=meta['connected_regions'],state=states[key]))
            for im in seedimages:im['seed']=seed
            for reg in seedregions:reg['seed']=seed
            all_images+=seedimages;all_regions+=seedregions;group=[]
            for name in repeat.NAMES:
                for g in 'ABCDE':
                    v=[r['PRO005'] for r in seedimages if r['method']==name and r['size_group']==g]
                    if not v:raise ValueError('FAIL_EMPTY_GROUP')
                    group.append(dict(method=name,size_group=g,n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),std=float(np.std(v))))
            groups[str(seed)]=group
            by={n:{r['source_id']:r['PRO005'] for r in seedimages if r['method']==n} for n in repeat.NAMES};ids=sorted(by['STD'])
            if set(ids)!=set(by['RECT']):raise ValueError('FAIL_PAIRED_IDS')
            paired_results[str(seed)]=paired([by['RECT'][i] for i in ids],[by['STD'][i] for i in ids])
            by={n:{(r['source_id'],r['region_id']):r['PRO005'] for r in seedregions if r['method']==n} for n in repeat.NAMES};s,r=by['STD'],by['RECT']
            if set(s)!=set(r):raise ValueError('FAIL_REGION_IDS')
            region_results[str(seed)]=dict(RECT_ONLY=sum(r[k]>0 and s[k]<=0 for k in s),STD_ONLY=sum(r[k]<=0 and s[k]>0 for k in s),BOTH=sum(r[k]>0 and s[k]>0 for k in s),BOTH_BAD=sum(r[k]<=0 and s[k]<=0 for k in s))
        base.csv_save(OUT/'predictions.csv',all_predictions);base.csv_save(OUT/'image_metrics.csv',all_images);base.csv_save(OUT/'region_metrics.csv',all_regions)
        if len(all_predictions)!=10040 or journal.counts['prediction']!=10040:raise ValueError('FAIL_PREDICTION_COUNT')
        runtime=benchmark(rows,guard,journal)
        summary=paper.summarize(data);dev=base.read(repeat.OUT/'summary/repeat_report.json')
        for k,v in binding['code'].items():
            if base.parent.digest(base.ROOT/k)!=v:raise ValueError('FAIL_CODE_CHANGED')
        if base.parent.digest(PROTOCOL)!=binding['protocol_sha']:raise ValueError('FAIL_PROTOCOL_CHANGED')
        for k,sha in EXPECTED_SHA.items():
            seed,name=k.split('_');verify_sha(checkpoint_path(int(seed),name),sha)
        result=dict(**FLAGS,status='PASS',metrics_by_seed=data,summary=summary,size_groups=paper.aggregate_groups(groups),size_groups_by_seed=groups,
            paired_by_seed=paired_results,regions_by_seed=region_results,runtime=runtime,state_integrity=states,training_cost=dev['training_cost'],
            input_cost=dict(STD=65536,RECT=276480,pixel_ratio=4.21875,parameters=8058628),dev_summary=dev['summary'],prediction_count=10040)
        base.csv_save(OUT/'paper_main_table.csv',paper.paper_rows(summary,runtime))
        base.csv_save(OUT/'per_seed_metrics.csv',[dict(seed=s,method=n,**data[str(s)][n]) for s in repeat.SEEDS for n in repeat.NAMES])
        base.csv_save(OUT/'per_seed_delta.csv',[dict(seed=s,**summary['per_seed_delta_pp'][str(s)]) for s in repeat.SEEDS])
        base.csv_save(OUT/'dev_vs_test_comparison.csv',[dict(Metric=k,Dev_STD=paper.display(dev['summary']['STD'][k]),Dev_RECT=paper.display(dev['summary']['RECT'][k]),Dev_Delta=paper.display(dev['summary']['delta_pp'][k],1),Test_STD=paper.display(summary['STD'][k]),Test_RECT=paper.display(summary['RECT'][k]),Test_Delta=paper.display(summary['delta_pp'][k],1)) for k in repeat.KEYS])
        journal.finish('PASS');result['audit']=journal.audit;base.save(OUT/'official_test_report.json',result)
        text=paper.text_report(result)+'\n\nFULL PRECISION RECORD\n'+json.dumps(result,indent=2)
        (OUT/'official_test_report.txt').write_text(text,encoding='utf-8');print(text,flush=True);return result
    except BaseException as exc:
        journal.finish('FAIL');base.save(OUT/'failure_report.json',dict(error=repr(exc),time=now()));raise
