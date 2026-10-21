"""Phase15D serial43-46 STD/RECT training; official test is never executed."""
import argparse
import gc
import sys
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from src.evaluation import ksdd2_repeat_summary as p
from src.training import ksdd2_repeat_training as t

def resume_smoke(seed,name,a,guard):
    m=t.fresh(seed,name);initial=t.old.branches(m);ds,order,_=t.setup(m,seed,name);t.seed_rng(seed);opt,sch=t.core.configure(m)
    seq=t.sequence(a,seed);lookup={r['id']:r for r in a['groups']['train_normal']};hw=tuple(p.config(seed,name)['input_hw'])
    def step(model,o,s,i):
        model.train();model.model.teacher.eval();x=t.old.normal_input(model,lookup[seq[i]]['image_path'],guard,'train');extra=t.old.extra_input(ds,order[i]);o.zero_grad(set_to_none=True)
        loss,aux,_=t.components(model,x,extra,hw,False)
        if aux is not None or not torch.isfinite(loss):raise ValueError('FAIL_SMOKE')
        loss.backward();o.step();s.step()
    for i in range(5):step(m,opt,sch,i)
    path=p.OUT/'logs'/f'seed{seed}_{name}_resume_smoke.pt';t.old.atomic_save(path,t.payload(m,opt,sch,seed,name,5))
    step(m,opt,sch,5);expected={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};after=t.old.branches(m)
    if initial['teacher']!=after['teacher'] or any(initial[k]==after[k] for k in ('student','ae')):raise ValueError('FAIL_SMOKE_UPDATES')
    del m,opt,sch;gc.collect();torch.cuda.empty_cache()
    m=t.fresh(seed,name);opt,sch=t.core.configure(m);c=torch.load(path,map_location='cpu',weights_only=False)
    if t.restore(m,opt,sch,c,seed,name)!=5:raise ValueError('FAIL_RESUME_STEP')
    step(m,opt,sch,5);error=t.core.continuation_error(m.state_dict(),expected)
    del m,opt,sch,c,ds;gc.collect();torch.cuda.empty_cache()
    return dict(status='PASS',saved_step=5,continued_step=6,exact_restored_state=True,continuation_max_abs_error=error)

def precheck():
    a=p.parent.assets();guard=p.base.AccessGuard(a);guard.install_test_block();references=t.references()
    p.development_assets(a,freeze=True)
    frozen=p.base.read(p.parent.OUT/'implementation_binding.json')
    files={**frozen,**{f:p.base.parent.digest(ROOT/f) for f in t.FILES}}
    t.core.freeze_json(p.OUT/'implementation_binding.json',files)
    t.core.freeze_json(p.OUT/'repeat_protocol.json',dict(**p.FLAGS,runs=[p.config(s,n) for s,n in p.RUNS],split_sha=p.base.SPLIT_SHA,
        seed42_reference_sha=p.REFERENCE_SHA,sequence_rule='per-seed permutation repeated; same sources for STD and RECT',all_eight_frozen_before_dev=True))
    for role in ('train_normal','validation_normal','excluded_train_defect'):
        for r in a['groups'][role]:
            if p.base.parent.digest(r['image_path'])!=r['image_sha']:raise ValueError('FAIL_DATA_CHANGED')
    hashes={};shapes={}
    for seed in p.SEEDS[1:]:
        t.sequence(a,seed);h={}
        for name in p.NAMES:
            m=t.fresh(seed,name);h[name]=t.old.branches(m);ds,order,paths=t.setup(m,seed,name)
            row=a['groups']['train_normal'][0];guard.check(row['image_path'],'train')
            shapes[p.key(seed,name)]=t.old.cal.shape_probe(m,row['image_path'],tuple(p.config(seed,name)['input_hw']))
            del m,ds,order,paths;gc.collect();torch.cuda.empty_cache()
        if h['STD']!=h['RECT']:raise ValueError('FAIL_WITHIN_SEED_INIT')
        hashes[str(seed)]=h['STD']
    all_hashes=[references['STD']['fresh_hashes']]+list(hashes.values())
    if len({h['teacher'] for h in all_hashes})!=1 or any(len({h[k] for h in all_hashes})!=5 for k in ('student','ae')):raise ValueError('FAIL_SEED_INDEPENDENCE')
    smoke={n:resume_smoke(43,n,a,guard) for n in p.NAMES}
    report=dict(**p.FLAGS,status='PRECHECK_PASS',canonical_hash=p.base.SPLIT_SHA,seed42=references,fresh_hashes=hashes,shapes=shapes,resume_smoke=smoke,access=guard.report())
    p.base.save(p.OUT/'precheck_report.json',report)
    print('Seed42 STD: FOUND / SHA PASS / strict reload PASS\nSeed42 RECT: FOUND / SHA PASS / strict reload PASS\nSeeds to train: 43,44,45,46\nMethods: STD + RECT only\nDGNB: DISABLED\nOfficial test: LOCKED\nEstimated additional training: ~54 h\nPRECHECK: PASS',flush=True)
    return a,guard,hashes,references

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--precheck-only',action='store_true');args=parser.parse_args()
    (p.OUT/'logs').mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);guard=None;current=None
    import msvcrt
    with (p.OUT/'process.lock').open('a+b') as lock:
        lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            a,guard,hashes,references=precheck()
            if args.precheck_only:return 0
            reports={}
            for seed,name in p.RUNS:
                current=p.key(seed,name);print('START '+current,flush=True)
                reports[current]=t.train_one(seed,name,a,guard,hashes[str(seed)])
                p.base.save(p.OUT/'training_progress.json',dict(current=current,reports=reports,access=guard.report()))
            p.all_ready(reports);p.run(a,guard,reports,references);return 0
        except Exception as ex:
            p.base.save(p.OUT/'failure_report.json',dict(**p.FLAGS,status='FAIL',current=current,error=str(ex),traceback=traceback.format_exc(),
                numerical_failure=isinstance(ex,(FloatingPointError,torch.cuda.OutOfMemoryError)),access=guard.report() if guard else None))
            traceback.print_exc();return 1

if __name__=='__main__':raise SystemExit(main())
