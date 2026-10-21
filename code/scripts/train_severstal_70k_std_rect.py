"""Phase16C sequential fresh70k and development-only audit."""
import argparse
import gc
import json
import sys
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.evaluation import severstal_70k_protocol as p
from src.training import severstal_formal_training as t

def resume_smoke(name,a,guard,seq):
    m=t.fresh(name);initial=t.old.branches(m);ds,order,_=t.setup(m,name);t.cal.seed();opt,sch=t.configure(m,a)
    m.train();m.model.teacher.eval();lookup={r['id']:r for r in a['groups']['train_normal']};hw=tuple(p.config(name)['input_hw'])
    def step(model,o,s,i):
        x=t.old.normal_input(model,lookup[seq[i]]['image_path'],guard,'train');extra=t.old.extra_input(ds,order[i]);o.zero_grad(set_to_none=True)
        loss,aux,_=t.cal.components(model,x,extra,hw,False)
        if aux is not None or not torch.isfinite(loss):raise ValueError('FAIL_RESUME_SMOKE_LOSS')
        loss.backward();o.step();s.step()
    for i in range(3):step(m,opt,sch,i)
    path=p.OUT/'logs'/f'{name}_resume_smoke.ckpt';t.old.atomic_save(path,t.payload(m,opt,sch,name,3))
    step(m,opt,sch,3);expected={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};after=t.old.branches(m)
    if after['teacher']!=initial['teacher'] or any(after[k]==initial[k] for k in ('student','ae')):raise ValueError('FAIL_RESUME_SMOKE_UPDATES')
    del m,opt,sch;gc.collect();torch.cuda.empty_cache()
    m=t.fresh(name);opt,sch=t.configure(m,a);c=torch.load(path,map_location='cpu',weights_only=False)
    if t.restore(m,opt,sch,c,name)!=3:raise ValueError('FAIL_RESTORE_STEP')
    m.train();m.model.teacher.eval();step(m,opt,sch,3);error=t.recovery.continuation_error(m.state_dict(),expected)
    if sch.last_epoch!=4:raise ValueError('FAIL_RESTORE_SCHEDULER')
    del m,opt,sch,ds,c;gc.collect();torch.cuda.empty_cache()
    return initial,dict(status='PASS',restored_step=3,continued_step=4,continuation_max_abs_error=error,Teacher_unchanged=True,Student_updated=True,AE_updated=True)

def precheck():
    a=p.assets();guard=p.base.Guard(a['rows']);guard.install();seq=p.sequence([r['id'] for r in a['groups']['train_normal']])
    t.freeze_csv(p.OUT/'train_sequence_manifest.csv',[dict(global_step=i+1,source_id=s) for i,s in enumerate(seq)])
    for role in ('train_normal','validation_normal','dev_anomaly'):
        for row in a['groups'][role]:
            if p.base.digest(row['image_path'])!=row['image_sha']:raise ValueError('FAIL_SOURCE_SHA')
            p.base.verify_mask(row)
    protocol=dict(FROZEN=True,stage='PHASE16C',methods={n:p.config(n) for n in p.NAMES},split_sha=p.SPLIT_SHA,geometry_sha=p.GEOMETRY_SHA,
        counts={k:len(v) for k,v in a['groups'].items()},size_boundaries=a['protocol']['size_boundaries'],pilot_report_sha=a['pilot_report_sha'],
        quantile_source='590 validation_normal only; unchanged official routine',decision_rule_source='frozen original Promising/Negative and user Strong',
        additional_report_labels='Await user-supplied prior frozen definitions; not used to change training',**p.FLAGS)
    t.freeze(p.OUT/'formal_dev_protocol.json',protocol)
    hashes={};smoke={}
    for n in p.NAMES:hashes[n],smoke[n]=resume_smoke(n,a,guard,seq)
    if hashes['STD']!=hashes['RECT']:raise ValueError('FAIL_FRESH_FAIRNESS')
    files=['src/evaluation/severstal_70k_protocol.py','src/training/severstal_formal_training.py','scripts/train_severstal_70k_std_rect.py',
        'src/training/severstal_r1_pilot.py','src/training/severstal_pilot.py','src/evaluation/severstal_protocol.py','src/evaluation/severstal_dev_evaluation.py',
        'src/training/ksdd2_pilot_training.py','src/training/ksdd2_formal_training.py','src/evaluation/ksdd2_training_calibration.py',
        'src/evaluation/ksdd2_dev_evaluation.py','src/evaluation/exact_aupro.py','src/evaluation/ksdd2_70k_protocol.py']
    t.freeze(p.OUT/'implementation_binding.json',{f:p.base.digest(ROOT/f) for f in files})
    report=dict(status='PASS',fresh_hashes=hashes,resume_continuity=smoke,split_sha=p.SPLIT_SHA,geometry_sha=p.GEOMETRY_SHA,**p.FLAGS)
    p.base.save(p.OUT/'precheck_report.json',report)
    print('PHASE16C PRECHECK PASS\nSplit SHA: PASS\nGeometry SHA: PASS\nSTD: fresh 256x256 seed42 FP32 batch1 70000\nRECT: fresh 256x1152 seed42 FP32 batch1 70000\nNormal quantile:590\nDEV:1923 after both finals\nFINAL BLIND:6514 LOCKED\nPilot checkpoint reuse:False\nDGNB/Attention/AMP/compile:DISABLED\nResume continuity:PASS',flush=True)
    return a,guard,seq,hashes['STD']

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--precheck-only',action='store_true');args=parser.parse_args()
    p.OUT.mkdir(parents=True,exist_ok=True);(p.OUT/'logs').mkdir(exist_ok=True);torch.set_num_threads(4)
    import msvcrt
    with (p.OUT/'execution.lock').open('a+b') as lock:
        lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            a,guard,seq,expected=precheck()
            if args.precheck_only:return 0
            reports={}
            for name in p.NAMES:
                reports[name]=t.train(name,a,guard,seq,expected)
                p.base.save(p.OUT/'training_progress.json',dict(completed=list(reports),reports=reports,**p.FLAGS))
            p.all_ready(reports);ep,et,e=t.evaluation_modules();result=e.run(a,guard,reports)
            result['decision_70k']=p.decide(result['metrics']['STD'],result['metrics']['RECT'])
            result['delta_10k_pp']=a['pilot']['decision']['delta_pp']
            result['additional_report_labels_status']='PENDING_FROZEN_DEFINITIONS'
            p.base.save(p.OUT/'phase16c_report.json',result);(p.OUT/'phase16c_report.txt').write_text(json.dumps(result,indent=2),encoding='utf-8')
            print(json.dumps(result,indent=2),flush=True);return 0
        except BaseException as exc:
            p.base.save(p.OUT/'failure_report.json',dict(status='FAIL',error=repr(exc),traceback=traceback.format_exc(),**p.FLAGS));raise

if __name__=='__main__':raise SystemExit(main())
