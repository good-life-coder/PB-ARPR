"""Frozen Phase16C policy; no geometry search, no blind model access."""
import copy
import numpy as np
from src.evaluation import severstal_protocol as base
from src.evaluation import ksdd2_70k_protocol as reference

OUT=base.ROOT/'experiments/phase16c_severstal_70k_formal_dev'
R1=base.ROOT/'experiments/phase16a_r1_severstal_geometry_revision'
PILOT=base.ROOT/'experiments/phase16b_r1_severstal_10k_pilot'
SPLIT_SHA='e38c69fa5287a619472b0afb97e5f33e32a95503369e065c096ffc1b44124594'
GEOMETRY_SHA='bce452c912826b119c237312997a4f57ea172afe6d1a09b3f2ca04952e8aba6d'
STEPS=70000
NAMES=('STD','RECT')
FOLDERS={'STD':'std_256_seed42','RECT':'rect_256x1152_seed42'}
FLAGS=dict(base.FLAGS,DEVELOPMENT_ONLY=True,PILOT_FOR_METHOD_DEVELOPMENT=False,
    REPEAT_COUNT=1,architecture_constrained=True,PILOT_CHECKPOINT_ALLOWED_FOR_FINAL=False,
    FINAL_BLIND_FORWARD_ALLOWED=False,ADDITIONAL_SEEDS_ALLOWED=False)

def config(name):
    if name not in NAMES:raise ValueError('FAIL_METHOD_DISABLED')
    c=reference.config(name)
    for k in reference.FLAGS:c.pop(k,None)
    c.update(FLAGS,stage='PHASE16C',input_hw=[256,256] if name=='STD' else [256,1152],
        method_description='Official EfficientAD-S baseline' if name=='STD' else 'User-proposed Architecture-Constrained Pixel-Budgeted Aspect-Ratio-Preserving EfficientAD-S',
        geometry_sha=GEOMETRY_SHA,split_sha=SPLIT_SHA)
    return c

def assets():
    a=copy.deepcopy(base.assets())
    if a['protocol']['canonical_split_sha256']!=SPLIT_SHA:raise ValueError('FAIL_SPLIT_SHA')
    counts={'train_normal':4131,'validation_normal':590,'dev_anomaly':1333,'final_blind_normal':1181,'final_blind_anomaly':5333}
    if {k:len(v) for k,v in a['groups'].items()}!=counts:raise ValueError('FAIL_COUNTS')
    path=R1/'geometry_protocol_r1.json'
    if base.digest(path)!=GEOMETRY_SHA:raise ValueError('FAIL_GEOMETRY_SHA')
    g=base.read(path)
    if not g['FROZEN'] or g['selected_hw']!=[256,1152] or not g['architecture_feasibility']:raise ValueError('FAIL_GEOMETRY_FREEZE')
    pilot=base.read(PILOT/'phase16_r1_report.json')
    if pilot['status']!='PASS' or not pilot['decision']['RECT_PROMISING']:raise ValueError('FAIL_PILOT_REFERENCE')
    a['geometry']=dict(g,rect_hw=g['selected_hw']);a['protocol']['geometry_sha256']=GEOMETRY_SHA
    a.update(methods={n:config(n) for n in NAMES},pilot=pilot,pilot_report_sha=base.digest(PILOT/'phase16_r1_report.json'))
    return a

def sequence(ids):
    order=np.random.default_rng(42).permutation(len(ids))
    return [ids[int(order[i%len(ids)])] for i in range(STEPS)]

def checkpoint_binding(c,name):
    if c.get('stage')!='PHASE16C' or c.get('config')!=config(name):raise ValueError('FAIL_PILOT_CONTAMINATION_OR_CONFIG')
    if c.get('method')!=name or c.get('input_hw')!=config(name)['input_hw']:raise ValueError('FAIL_METHOD_GEOMETRY')
    if c.get('split_sha')!=SPLIT_SHA or c.get('geometry_sha')!=GEOMETRY_SHA or not 0<=c.get('global_step',-1)<=STEPS:raise ValueError('FAIL_CHECKPOINT_BINDING')
    return True

def final_gate(c,name):
    checkpoint_binding(c,name)
    if c.get('global_step')!=STEPS or c.get('validation_count')!=590:raise ValueError('FAIL_FINAL_VALIDATION')
    q=c.get('quantiles',{})
    if set(q)!={'qa_st','qb_st','qa_ae','qb_ae'} or not all(np.isfinite(v) for v in q.values()) or any(q['qa_'+k]>=q['qb_'+k] for k in ('st','ae')):raise ValueError('FAIL_QUANTILES')
    return True

all_ready=reference.all_ready

def decide(s,r):
    d=base.decide(s,r)
    return dict(delta_pp=d['delta_pp'],**{k.replace('RECT_','RECT_70K_'):d[k] for k in ('RECT_PROMISING','RECT_STRONG','RECT_NEGATIVE','RECT_INCONCLUSIVE')})
