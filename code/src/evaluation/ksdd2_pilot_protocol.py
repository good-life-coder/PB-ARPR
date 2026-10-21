"""Immutable Phase15B roles and decisions; official test is metadata-only."""
import copy
import csv
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from src.evaluation import ksdd2_external_protocol as parent

ROOT=parent.ROOT
OUT=ROOT/'experiments/phase15b_ksdd2_10k_pilot'
PARENT_SHA='4dbe1817534ff41eee134206a8d76da1d61739c159e72f2ca3d0deb545233176'
SPLIT_SHA='6ebea6a7a2628697627a31e45489904fa7ec61762ef06eef8ec54c77c3a1b3b7'
NAMES=('STD','RECT','DGNB')
STEPS=10000
FOLDERS=dict(STD='std_256_seed42',RECT='rect_864x320_seed42',DGNB='dgnb_864x320_seed42')
FLAGS=dict(PILOT_STEPS=10000,FINAL_TRAINING_STEPS=70000,PILOT_FOR_METHOD_DEVELOPMENT=True,
 DO_NOT_REPORT_AS_FINAL_KSDD2_RESULT=True,DEV_NORMAL_REUSED_FOR_QUANTILE=True,
 PILOT_CHECKPOINT_ALLOWED_FOR_FINAL=False,FINAL_KSDD2_TEST_UNTOUCHED=True)

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
def csv_save(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        if rows:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def assets():
    pp=parent.OUT/'external_protocol.json';report=read(parent.OUT/'phase15a_report.json')
    if parent.digest(pp)!=PARENT_SHA or report['status']!='PASS':raise ValueError('FAIL_PARENT_PROTOCOL')
    with (parent.OUT/'split_manifest.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
    if parent.split_hash(rows)!=SPLIT_SHA:raise ValueError('FAIL_SPLIT_HASH')
    expected=dict(train_normal=1876,validation_normal=209,excluded_train_defect=246,test=1004)
    if Counter(r['split'] for r in rows)!=expected:raise ValueError('FAIL_SPLIT_COUNTS')
    groups={role:sorted([r for r in rows if r['split']==role],key=lambda r:r['id']) for role in expected}
    for role,label in [('train_normal','0'),('validation_normal','0'),('excluded_train_defect','1')]:
        if any(r['label']!=label or r['official_split']!='train' for r in groups[role]):raise ValueError('FAIL_ROLE')
    if Counter(r['label'] for r in groups['test'])!={'0':894,'1':110}:raise ValueError('FAIL_TEST_METADATA')
    if parent.duplicate_audit(rows)['split_leakage']:raise ValueError('FAIL_LEAKAGE')
    bypath={str(Path(r[k]).resolve()).casefold():r for r in rows for k in ('image_path','mask_path')}
    return dict(parent=read(pp),parent_sha=PARENT_SHA,split_sha=SPLIT_SHA,groups=groups,rows=rows,bypath=bypath)

def pilot_protocol(a):
    return dict(**FLAGS,parent_protocol=str(parent.OUT/'external_protocol.json'),parent_sha=PARENT_SHA,
        split_sha=SPLIT_SHA,pilot_steps=STEPS,seed=42,methods=copy.deepcopy(a['parent']['methods']),
        dev_anomaly_source='246 excluded official-train defects; permanent DEV_ANOMALY',
        dev_normal_source='209 validation normal',official_test_used=False,formal_test_tuning_allowed=False,
        pilot_results_are_final=False,formal_training_must_be_fresh=True,
        size_groups=copy.deepcopy(a['parent']['size_groups']),size_groups_used_for_decision=False,
        sequence_rule='seed42 permutation repeated, matching Phase15A ordering principle',
        imagenette_sequence='seed42 permutation repeated; official ImageFolder and transforms retained',
        scheduler='parent frozen StepLR66500 gamma0.1; no pilot schedule compression',
        probes=[200,1000,5000,10000],probe_samples=16,metric_implementation='src.evaluation.exact_aupro',
        native_exact=True,all_three_frozen_before_dev=True)

def sequence(ids):
    order=np.random.default_rng(42).permutation(len(ids))
    return [ids[int(order[i%len(ids)])] for i in range(STEPS)]

def authorize(a,path,purpose,dev_ready):
    key=str(Path(path).resolve()).casefold();row=a['bypath'].get(key)
    if row is None:raise ValueError('FAIL_UNREGISTERED_SOURCE '+str(path))
    role=row['split']
    if role=='test':raise ValueError('FAIL_OFFICIAL_TEST_ACCESS')
    allowed={'train':{'train_normal'},'probe':{'train_normal'},'quantile':{'validation_normal'},
        'dev':{'validation_normal','excluded_train_defect'}}
    if role not in allowed.get(purpose,set()) or (purpose=='dev' and not dev_ready):raise ValueError('FAIL_SOURCE_ROLE')
    return row['id']

class AccessGuard:
    def __init__(self,a):self.a=a;self.ready=False;self.counts=Counter();self.blocked=0
    def check(self,path,purpose):
        sid=authorize(self.a,path,purpose,self.ready);self.counts[purpose]+=1;return sid
    def install_test_block(self):
        roots={str(Path(r['image_path']).parent.resolve()).casefold() for r in self.a['groups']['test']}
        def audit(event,args):
            if event=='open' and isinstance(args[0],(str,bytes,Path)):
                path=str(Path(args[0]).resolve()).casefold()
                if any(path==root or path.startswith(root+'\\') or path.startswith(root+'/') for root in roots):
                    self.blocked+=1;raise PermissionError('FAIL_OFFICIAL_TEST_FILE_ACCESS')
        sys.addaudithook(audit)
    def report(self):return dict(allowed_reads=dict(self.counts),blocked_test_access_attempts=self.blocked,
        OFFICIAL_TEST_FORWARD_COUNT=0,OFFICIAL_TEST_PREDICTIONS_CREATED=0,FINAL_KSDD2_TEST_UNTOUCHED=self.blocked==0)

def all_ready(reports):
    if set(reports)!=set(NAMES) or any(r.get('status')!='PASS' or r.get('completed')!=STEPS or not r.get('strict_reload') or not r.get('quantiles_frozen') for r in reports.values()):
        raise ValueError('FAIL_ALL_PILOTS_NOT_FROZEN')
    return True

def checkpoint_gate(c):
    if c['global_step']!=STEPS or c['validation_count']!=209:raise ValueError('FAIL_CHECKPOINT_STEP_OR_VALIDATION')
    keys={'qa_st','qb_st','qa_ae','qb_ae'}|({'qa_s1','qb_s1'} if c['method']=='DGNB' else set())
    if set(c['quantiles'])!=keys or not all(np.isfinite(v) for v in c['quantiles'].values()):raise ValueError('FAIL_QUANTILES')
    for tag in ('st','ae')+(('s1',) if c['method']=='DGNB' else ()):
        if c['quantiles']['qb_'+tag]<=c['quantiles']['qa_'+tag]:raise ValueError('FAIL_QUANTILE_RANGE')
    return True

def checkpoint_binding(c,name,hw):
    expected={'STD':(256,256),'RECT':(864,320),'DGNB':(864,320)}
    if c.get('method')!=name or tuple(c.get('input_hw',()))!=expected[name] or tuple(hw)!=expected[name]:
        raise ValueError('FAIL_CHECKPOINT_METHOD_GEOMETRY_BINDING')
    return True

def native(t,hw):
    a=torch.nn.functional.interpolate(t.detach().float().cpu(),size=tuple(hw),mode='bilinear',align_corners=False)[0,0].numpy()
    if not np.isfinite(a).all():raise ValueError('FAIL_NONFINITE_MAP')
    return a

def prediction_gate(a,rows):
    official={r['id'] for r in a['groups']['test']};dev={r['id'] for k in ('validation_normal','excluded_train_defect') for r in a['groups'][k]}
    for r in rows:
        sid=r.get('source_id',r.get('sample_id'))
        if sid in official or sid not in dev:raise ValueError('FAIL_TEST_CONTAMINATION')
    return True

def decide(m):
    s,r,d=(m[n] for n in NAMES);eps=1e-12
    ge=lambda x,y:x>=y-eps
    le=lambda x,y:x<=y+eps
    rp=(ge(r['aupro005'],s['aupro005']+.01) or (ge(r['pixel_auroc'],s['pixel_auroc']+.01) and ge(r['aupro005'],s['aupro005']-.005))) and ge(r['image_auroc'],s['image_auroc']-.05)
    rn=not rp and le(r['aupro005'],s['aupro005']-.01) and le(r['pixel_auroc'],s['pixel_auroc'])
    dp=(ge(d['aupro005'],r['aupro005']+.01) and ge(d['image_auroc'],r['image_auroc']-.05)) or (ge(d['aupro005'],s['aupro005']+.015) and ge(d['image_auroc'],s['image_auroc']-.05))
    dn=not dp and le(d['aupro005'],r['aupro005']-.01) and le(d['pixel_auroc'],r['pixel_auroc']) and le(d['image_auroc'],r['image_auroc'])
    justified=rp or dp
    return dict(RECT_PROMISING=rp,RECT_NEGATIVE=rn,RECT_INCONCLUSIVE=not(rp or rn),
        DGNB_PROMISING=dp,DGNB_NEGATIVE=dn,DGNB_INCONCLUSIVE=not(dp or dn),
        FORMAL_EXTERNAL_TRAINING_JUSTIFIED=justified,RECOMMEND_STD_70K=justified,
        RECOMMEND_RECT_70K=justified,RECOMMEND_DGNB_70K=dp,
        NEXT_ACTION='FRESH_FORMAL_70K_RECOMMENDED_NOT_STARTED' if justified else 'STOP_KSDD2_FORMAL_AND_ANALYZE_DOMAIN_DIFFERENCE' if rn and dn else 'EXTEND_INCONCLUSIVE_PILOT_TO_20K',
        PILOT_CHECKPOINT_ALLOWED_FOR_FINAL=False)
