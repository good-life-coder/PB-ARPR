"""Phase15C policy: STD/RECT only, fresh70000, official test locked."""
import numpy as np
from src.evaluation import ksdd2_pilot_protocol as base

OUT=base.ROOT/'experiments/phase15c_ksdd2_70k_formal_dev'
NAMES=('STD','RECT')
STEPS=70000
FOLDERS={n:base.FOLDERS[n] for n in NAMES}
FLAGS=dict(PROTOCOL='Development verification before blind test',REPEAT_COUNT=1,
    PAPER_FINAL_REQUIRES_REPEAT_ANALYSIS=True,DEV_NORMAL_REUSED_FOR_QUANTILE=True,
    PILOT_CHECKPOINT_ALLOWED_FOR_FINAL=False,FINAL_KSDD2_TEST_UNTOUCHED=True,
    OFFICIAL_TEST_FORWARD_COUNT=0,OFFICIAL_TEST_PREDICTIONS_CREATED=0,DGNB_DISABLED=True,
    DEVELOPMENT_ONLY=True,DO_NOT_REPORT_AS_FINAL_KSDD2_TEST_RESULT=True)

def config(name):
    if name not in NAMES:raise ValueError('FAIL_METHOD_DISABLED')
    return dict(**FLAGS,stage='PHASE15C',method=name,
        method_description='Official EfficientAD-S baseline' if name=='STD' else 'User-proposed aspect-ratio-preserving EfficientAD-S',
        input_hw=[256,256] if name=='STD' else [864,320],steps=STEPS,max_epochs=1000,
        batch=1,precision='32-true',seed=42,parameters=8058628,new_parameters=0,
        model_size='small',fresh_student=True,fresh_ae=True,teacher='official pretrained',
        checkpoint_initialization=None,early_stopping=False,amp=False,torch_compile=False,
        optimizer='official Adam',lr=1e-4,weight_decay=1e-5,scheduler_step=66500,scheduler_gamma=.1,
        loss='unchanged official EfficientAD ST + AE + STAE including one ImageNette penalty',
        image_head='official Combined native max',pixel_head='official Combined',
        official_test_forward_allowed=False)

def assets():
    a=base.assets();path=base.OUT/'dev_evaluation/dev_report.json';pilot=base.read(path)
    if pilot['status']!='PASS' or not pilot['decision']['RECT_PROMISING'] or not pilot['decision']['DGNB_NEGATIVE'] or pilot['decision']['RECOMMEND_DGNB_70K']:
        raise ValueError('FAIL_PILOT_DECISION')
    a.update(pilot=pilot,pilot_report_sha=base.parent.digest(path));return a

def sequence(ids):
    order=np.random.default_rng(42).permutation(len(ids))
    return [ids[int(order[i%len(ids)])] for i in range(STEPS)]

def checkpoint_binding(c,name):
    if c.get('stage')!='PHASE15C' or 'PILOT_STEPS' in c:raise ValueError('FAIL_PILOT_CONTAMINATION')
    if name not in NAMES or c.get('method')!=name or c.get('config')!=config(name) or c.get('input_hw')!=config(name)['input_hw']:
        raise ValueError('FAIL_CHECKPOINT_CONFIG_BINDING')
    if c.get('parent_sha')!=base.PARENT_SHA or c.get('split_sha')!=base.SPLIT_SHA or not 0<=c.get('global_step',-1)<=STEPS:
        raise ValueError('FAIL_CHECKPOINT_STEP_OR_PROTOCOL')
    return True

def final_gate(c,name):
    checkpoint_binding(c,name)
    if c.get('global_step')!=STEPS or c.get('validation_count')!=209:raise ValueError('FAIL_FINAL_VALIDATION')
    q=c.get('quantiles',{})
    if set(q)!={'qa_st','qb_st','qa_ae','qb_ae'} or not all(np.isfinite(v) for v in q.values()):raise ValueError('FAIL_QUANTILES')
    if any(q['qb_'+tag]<=q['qa_'+tag] for tag in ('st','ae')):raise ValueError('FAIL_QUANTILE_RANGE')
    return True

def all_ready(reports):
    if set(reports)!=set(NAMES) or any(r.get('status')!='PASS' or r.get('completed')!=STEPS or
        not all(r.get(k) for k in ('strict_reload','quantiles_reload','teacher_unchanged')) or
        any(r.get(k,True) for k in ('NaN','Inf','OOM')) for r in reports.values()):
        raise ValueError('FAIL_BOTH_FINALS_NOT_FROZEN')
    return True

def decide(m,pilot,integrity=True,test_untouched=True):
    s,r=m['STD'],m['RECT'];eps=1e-12
    ge=lambda a,b:a>=b-eps
    le=lambda a,b:a<=b+eps
    promising=(ge(r['aupro005'],s['aupro005']+.01) or (ge(r['pixel_auroc'],s['pixel_auroc']+.01) and ge(r['aupro005'],s['aupro005']-.005))) and ge(r['image_auroc'],s['image_auroc']-.05)
    strong=ge(r['aupro005'],s['aupro005']+.03) and ge(r['pixel_auroc'],s['pixel_auroc']+.01) and ge(r['image_auroc'],s['image_auroc']-.02)
    negative=not promising and le(r['aupro005'],s['aupro005']-.01) and le(r['pixel_auroc'],s['pixel_auroc'])
    keys=('image_auroc','pixel_auroc','aupro005','aupro030')
    delta={k:100*(r[k]-s[k]) for k in keys};old={k:100*(pilot['RECT'][k]-pilot['STD'][k]) for k in keys}
    ready=bool(promising and integrity and test_untouched)
    return dict(RECT_70K_PROMISING=bool(promising),RECT_70K_STRONG=bool(strong),RECT_70K_NEGATIVE=bool(negative),
        RECT_70K_INCONCLUSIVE=not(promising or negative),TREND_PERSISTED=delta['aupro005']>0 and delta['pixel_auroc']>0,
        delta_70k_pp=delta,delta_10k_pp=old,READY_FOR_ONE_SHOT_OFFICIAL_TEST=ready,
        NEXT_ACTION='AWAIT_ONE_SHOT_OFFICIAL_TEST_AUTHORIZATION' if ready else 'STOP_BEFORE_OFFICIAL_TEST')
