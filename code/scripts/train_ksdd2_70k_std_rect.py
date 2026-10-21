"""Phase15C: fresh STD -> RECT70000, normal validation, development only."""
import argparse
import gc
import json
import sys
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from src.evaluation import ksdd2_70k_protocol as p
from src.training import ksdd2_formal_training as t

def real_resume_check(name,a,guard,seq,initial):
    m=t.fresh(name);ds,order,_=t.setup(m,name);t.old.cal.seed();opt,sch=t.configure(m);m.train();m.model.teacher.eval()
    lookup={r['id']:r for r in a['groups']['train_normal']};hw=tuple(p.config(name)['input_hw'])
    def step(model,o,s,i):
        x=t.old.normal_input(model,lookup[seq[i]]['image_path'],guard,'train');extra=t.old.extra_input(ds,order[i]);o.zero_grad(set_to_none=True)
        loss,_,_=t.components(model,x,extra,hw,False)
        if not torch.isfinite(loss):raise ValueError('FAIL_SMOKE_LOSS')
        loss.backward();o.step();s.step()
    for i in range(5):step(m,opt,sch,i)
    path=p.OUT/'logs'/f'{name}_resume_smoke.pt';t.old.atomic_save(path,t.resume_payload(m,opt,sch,name,5))
    step(m,opt,sch,5);expected={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};after=t.old.branches(m)
    if after['teacher']!=initial['teacher'] or any(after[k]==initial[k] for k in ('student','ae')):raise ValueError('FAIL_SMOKE_UPDATES')
    del m,opt,sch;gc.collect();torch.cuda.empty_cache()
    m=t.fresh(name);opt,sch=t.configure(m);c=torch.load(path,map_location='cpu',weights_only=False)
    if t.restore(m,opt,sch,c,name)!=5:raise ValueError('FAIL_RESUME_STEP')
    m.train();m.model.teacher.eval();step(m,opt,sch,5)
    error=t.continuation_error(m.state_dict(),expected)
    if sch.last_epoch!=6:raise ValueError('FAIL_RESUME_CONTINUITY')
    del m,opt,sch,c,ds;gc.collect();torch.cuda.empty_cache()
    return dict(status='PASS',saved_step=5,restored_step=5,continued_step=6,restored_model_optimizer_scheduler_exact=True,
        continuation_max_abs_error=error,continuation_atol=torch.finfo(torch.float32).eps,teacher_unchanged=True)

def precheck():
    a=p.assets();guard=p.base.AccessGuard(a);guard.install_test_block();seq=t.freeze_sequences(a)
    protocol=dict(**p.FLAGS,stage='PHASE15C',parent_protocol=str(p.base.parent.OUT/'external_protocol.json'),
        parent_sha=p.base.PARENT_SHA,split_sha=p.base.SPLIT_SHA,pilot_report_sha=a['pilot_report_sha'],
        methods={n:p.config(n) for n in p.NAMES},size_groups=a['parent']['size_groups'],size_groups_used_for_decision=False,
        metric_implementation='Phase15B exact rank-based native AUPRO',runtime_test_execution=False)
    t.freeze_json(p.OUT/'formal_dev_protocol.json',protocol)
    files=['src/evaluation/ksdd2_70k_protocol.py','src/training/ksdd2_formal_training.py','src/evaluation/ksdd2_70k_dev_evaluation.py',
        'scripts/train_ksdd2_70k_std_rect.py','scripts/evaluate_ksdd2_70k_dev.py','src/training/ksdd2_pilot_training.py',
        'src/evaluation/ksdd2_training_calibration.py','src/evaluation/ksdd2_pilot_protocol.py','src/evaluation/ksdd2_external_protocol.py',
        'src/evaluation/ksdd2_dev_evaluation.py','src/evaluation/exact_aupro.py']
    t.freeze_json(p.OUT/'implementation_binding.json',{f:p.base.parent.digest(ROOT/f) for f in files})
    for role in ('train_normal','validation_normal','excluded_train_defect'):
        for r in a['groups'][role]:
            if p.base.parent.digest(r['image_path'])!=r['image_sha']:raise ValueError('FAIL_DATA_CHANGED')
    hashes={};shapes={};resumes={}
    for n in p.NAMES:
        m=t.fresh(n);hashes[n]=t.old.branches(m);row=a['groups']['train_normal'][0];guard.check(row['image_path'],'train')
        shapes[n]=t.old.cal.shape_probe(m,row['image_path'],tuple(p.config(n)['input_hw']))
        del m;gc.collect();torch.cuda.empty_cache()
        resumes[n]=real_resume_check(n,a,guard,seq,hashes[n])
    if hashes['STD']!=hashes['RECT']:raise ValueError('FAIL_FRESH_HASHES')
    r=dict(**p.FLAGS,status='PRECHECK_PASS',canonical_hash=p.base.SPLIT_SHA,fresh_hashes=hashes,shapes=shapes,
        resume_continuity=resumes,train_normal=1876,validation_normal=209,dev_defect=246,official_test=1004,access=guard.report())
    p.base.save(p.OUT/'precheck_report.json',r)
    print('Phase15A split: PASS\nCanonical hash: PASS\nSTD: fresh / 256x256 / 70k\nRECT: fresh / 864x320 / 70k\nDGNB: DISABLED\nDev anomalies: 246\nOfficial test: LOCKED\nPilot checkpoint reuse: False\nResume continuity: PASS\nExpected STD: ~3.2 h\nExpected RECT: ~10.6 h\nExpected total: ~14-15 h\nPRECHECK: PASS',flush=True)
    return a,guard,seq,hashes['STD']

def main():
    args=argparse.ArgumentParser();args.add_argument('--precheck-only',action='store_true');arg=args.parse_args()
    p.OUT.mkdir(parents=True,exist_ok=True);(p.OUT/'logs').mkdir(exist_ok=True);torch.set_num_threads(4);guard=None
    # OS-owned lock is released on interruption; no stale PID file can trigger parallel training.
    import msvcrt
    with (p.OUT/'process.lock').open('a+b') as lock:
        lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            a,guard,seq,initial=precheck()
            if arg.precheck_only:return 0
            reports={}
            for n in p.NAMES:
                reports[n]=t.train_one(n,a,guard,seq,initial)
                p.base.save(p.OUT/'training_progress.json',dict(**p.FLAGS,reports=reports,completed=list(reports),access=guard.report()))
            p.all_ready(reports)
            from src.evaluation.ksdd2_70k_dev_evaluation import run
            run(a,guard,reports);return 0
        except Exception as ex:
            p.base.save(p.OUT/'failure_report.json',dict(**p.FLAGS,status='FAIL',error=str(ex),traceback=traceback.format_exc(),access=guard.report() if guard else None))
            traceback.print_exc();return 1

if __name__=='__main__':raise SystemExit(main())
