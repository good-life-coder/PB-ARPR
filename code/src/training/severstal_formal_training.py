"""Phase16C orchestration; unchanged official model/loss/normal quantiles."""
import csv
import gc
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from src.evaluation import severstal_70k_protocol as p
from src.training import severstal_pilot as pilot
from src.training import ksdd2_formal_training as recovery
from src.training import ksdd2_pilot_training as old
from src.evaluation import ksdd2_training_calibration as cal

def clone(name,path):
    spec=importlib.util.spec_from_file_location(name,p.base.ROOT/path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def freeze(path,value):
    if path.exists() and p.base.read(path)!=value:raise ValueError('FAIL_FROZEN_CHANGED '+str(path))
    if not path.exists():p.base.save(path,value)

def freeze_csv(path,rows):
    if path.exists():
        with path.open(encoding='utf-8') as f:prior=list(csv.DictReader(f))
        if prior!=[{k:str(v) for k,v in r.items()} for r in rows]:raise ValueError('FAIL_SEQUENCE_CHANGED')
    else:p.base.csv_save(path,rows)

def fresh(name):return cal.fresh(tuple(p.config(name)['input_hw']))

def configure(m,a):
    loader=torch.utils.data.DataLoader(a['groups']['train_normal'],batch_size=1,shuffle=False)
    m._trainer=SimpleNamespace(max_epochs=1000,max_steps=70000,datamodule=SimpleNamespace(train_dataloader=lambda:loader))
    c=m.configure_optimizers();opt,sch=c['optimizer'],c['lr_scheduler']
    if sch.step_size!=66500 or sch.gamma!=.1:raise ValueError('FAIL_SCHEDULER')
    return opt,sch

def setup(m,name):
    path=p.R1/f'teacher_stats_{name}.pt'
    if p.base.digest(path)!=p.base.read(p.R1/'model_precheck.json')['teacher_stats_sha'][name]:raise ValueError('FAIL_TEACHER_STATS_SHA')
    stats=torch.load(path,map_location='cuda',weights_only=True);old.validate_teacher_stats(stats);m.model.mean_std.update(stats)
    cal.prepare_extra(m,tuple(p.config(name)['input_hw']));ds=m.imagenet_loader.dataset
    seq=p.sequence(list(range(len(ds))));paths=[str(Path(ds.samples[i][0]).resolve()) for i in seq]
    freeze_csv(p.OUT/'imagenette_sequence_manifest.csv',[dict(global_step=i+1,source_path=s) for i,s in enumerate(paths)])
    return ds,seq,paths

def payload(m,opt,sch,name,step,**extra):
    return dict(stage='PHASE16C',config=p.config(name),method=name,input_hw=p.config(name)['input_hw'],
        split_sha=p.SPLIT_SHA,geometry_sha=p.GEOMETRY_SHA,global_step=step,model_state=m.state_dict(),
        optimizer=opt.state_dict(),scheduler=sch.state_dict(),rng=recovery.rng_state(),**extra)

def restore(m,opt,sch,c,name):
    module=clone('src.training._phase16c_restore','src/training/ksdd2_formal_training.py');module.p=p
    return module.restore(m,opt,sch,c,name)

def assert_final_path(path,name):
    if Path(path).resolve()!=(p.OUT/p.FOLDERS[name]/'final.ckpt').resolve():raise ValueError('FAIL_PILOT_CONTAMINATION')

def load_final(path,name):
    assert_final_path(path,name);c=torch.load(path,map_location='cpu',weights_only=False);p.final_gate(c,name)
    m=fresh(name);m.load_state_dict(c['model_state'],strict=True);m.eval()
    if sum(x.numel() for x in m.parameters())!=8058628 or old.state_hash(m)!=c['state_hash'] or old.quantile_values(m,name)!=c['quantiles']:raise ValueError('FAIL_STRICT_RELOAD')
    return m,c

def load_for_eval(name,a):
    path=p.OUT/p.FOLDERS[name]/'final.ckpt';report=p.base.read(path.parent/'training_report.json')
    if p.base.digest(path)!=report['checkpoint_sha256']:raise ValueError('FAIL_CHECKPOINT_SHA')
    return load_final(path,name)

def bound(a):
    code=p.base.read(p.OUT/'implementation_binding.json')
    if any(p.base.digest(p.base.ROOT/f)!=h for f,h in code.items()):raise ValueError('FAIL_CODE_CHANGED')
    return dict(code=code,split_sha=p.SPLIT_SHA,geometry_sha=p.GEOMETRY_SHA,pilot_report_sha=a['pilot_report_sha'],
        train_sequence_sha=p.base.digest(p.OUT/'train_sequence_manifest.csv'),ImageNette_sequence_sha=p.base.digest(p.OUT/'imagenette_sequence_manifest.csv'))

def save_report(path,c):
    r=dict(c['training_summary'],checkpoint=str(path.resolve()),checkpoint_sha256=p.base.digest(path),checkpoint_size=path.stat().st_size,
        strict_reload=True,quantiles_reload=True)
    p.base.save(path.parent/'training_report.json',r);(path.parent/'training_report.txt').write_text(json.dumps(r,indent=2),encoding='utf-8');return r

def train(name,a,guard,seq,expected):
    folder=p.OUT/p.FOLDERS[name];(folder/'checkpoints').mkdir(parents=True,exist_ok=True);(folder/'logs').mkdir(exist_ok=True)
    freeze(folder/'config.json',p.config(name));last=folder/'checkpoints/last.ckpt';final=folder/'final.ckpt'
    if final.exists():
        m,c=load_final(final,name)
        if c['bindings']!=bound(a):raise ValueError('FAIL_FINAL_BINDINGS')
        r=save_report(final,c);del m,c;gc.collect();torch.cuda.empty_cache();return r
    m=fresh(name);initial=old.branches(m)
    if initial!=expected:raise ValueError('FAIL_FRESH_INITIALIZATION')
    ds,order,paths=setup(m,name);cal.seed();opt,sch=configure(m,a);binding=bound(a)
    history=[];elapsed=0.;peak_a=peak_r=0.;resumes=[];step=0
    if last.exists():
        c=torch.load(last,map_location='cpu',weights_only=False);p.checkpoint_binding(c,name)
        if c.get('bindings')!=binding or c.get('fresh_hashes')!=initial:raise ValueError('FAIL_RESUME_BINDINGS')
        step=restore(m,opt,sch,c,name);history=c['history'];elapsed=c['training_seconds'];resumes=c['resume_steps']+[step]
        peak_a=c['peak_allocated_mib'];peak_r=c['peak_reserved_mib']
        if len(history)!=step or any(r['global_step']!=i+1 or r['source_id']!=seq[i] or r['imagenette_source']!=paths[i] for i,r in enumerate(history)):raise ValueError('FAIL_RESUME_HISTORY')
        del c
    elif (folder/'loss_history.csv').exists():raise ValueError('FAIL_HISTORY_WITHOUT_CHECKPOINT')
    lookup={r['id']:r for r in a['groups']['train_normal']};hw=tuple(p.config(name)['input_hw']);m.train();m.model.teacher.eval()
    def save_last(n,seconds):
        old.atomic_save(last,payload(m,opt,sch,name,n,history=history,training_seconds=seconds,fresh_hashes=initial,bindings=binding,
            resume_steps=resumes,peak_allocated_mib=peak_a,peak_reserved_mib=peak_r))
    if not last.exists():save_last(0,0.)
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    with (folder/'loss_history.csv').open('w',newline='',encoding='utf-8') as f:
        writer=None
        if history:writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history);f.flush()
        for i in range(step,70000):
            torch.cuda.synchronize();begin=time.perf_counter();x=old.normal_input(m,lookup[seq[i]]['image_path'],guard,'train');extra=old.extra_input(ds,order[i]);opt.zero_grad(set_to_none=True)
            total,aux,row=cal.components(m,x,extra,hw,False)
            if aux is not None or not torch.isfinite(total):raise FloatingPointError('TRAINING_CRITICAL_FAIL_LOSS')
            total.backward()
            if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in m.parameters()):raise FloatingPointError('TRAINING_CRITICAL_FAIL_GRADIENT')
            opt.step();sch.step();torch.cuda.synchronize()
            row.update(global_step=i+1,source_id=seq[i],imagenette_source=paths[i],total_loss=float(total.detach()),lr=opt.param_groups[0]['lr'],seconds=time.perf_counter()-begin)
            history.append(row)
            if writer is None:writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader()
            writer.writerow(row);del x,extra,total,aux
            if (i+1)%100==0:f.flush();print(f'{name} {i+1}/70000 | sec/step={np.mean([r["seconds"] for r in history[-100:]]):.4f} | lr={row["lr"]}',flush=True)
            if (i+1)%500==0:
                peak_a=max(peak_a,torch.cuda.max_memory_allocated()/2**20);peak_r=max(peak_r,torch.cuda.max_memory_reserved()/2**20);save_last(i+1,elapsed+time.perf_counter()-start)
    elapsed+=time.perf_counter()-start;after=old.branches(m)
    if after['teacher']!=initial['teacher'] or any(after[k]==initial[k] for k in ('student','ae')):raise ValueError('FAIL_BRANCH_UPDATE')
    opt.zero_grad(set_to_none=True);q=old.validation(m,name,a,guard)
    summary=dict(**p.config(name),status='PASS',completed=70000,validation_count=590,quantiles=q,quantiles_frozen=True,
        training_hours=elapsed/3600,seconds_per_step=float(np.mean([r['seconds'] for r in history[50:]])),peak_allocated_mib=peak_a,peak_reserved_mib=peak_r,
        fresh_hashes=initial,final_hashes=after,teacher_unchanged=True,student_updated=True,ae_updated=True,NaN=False,Inf=False,OOM=False,
        resume_used=bool(resumes),resume_from_steps=resumes,bindings=binding,scheduler_final=sch.state_dict())
    c=payload(m,opt,sch,name,70000,validation_count=590,quantiles=q,state_hash=old.state_hash(m),bindings=binding,training_summary=summary)
    p.final_gate(c,name);old.atomic_save(final,c);restored,loaded=load_final(final,name)
    if old.state_hash(restored)!=old.state_hash(m):raise ValueError('FAIL_FINAL_RELOAD')
    r=save_report(final,c);print(f'{name} 70000/70000 +590normal validation +strict reload PASS; SHA={r["checkpoint_sha256"]}',flush=True)
    del m,opt,sch,ds,c,restored,loaded;gc.collect();torch.cuda.empty_cache();return r

def evaluation_modules():
    from src.training.severstal_r1_pilot import isolated_modules
    ep,et,e=isolated_modules();ep.PILOT=p.OUT;ep.FLAGS=dict(p.FLAGS);et.load_final=load_for_eval;et.all_ready=p.all_ready
    def decision(s,r):
        d=p.base.decide(s,r)
        for k in ('RECOMMEND_70K','STOP_SEVERSTAL_RECT_LINE','NEXT_ACTION'):d.pop(k)
        return d
    ep.decide=decision
    # Reuse the frozen evaluation and runtime bodies; geometry selection is never called here.
    # Its optional future70k draft is inappropriate after70k; keep decision reporting but suppress only that draft writer.
    save=ep.save
    def save_eval(path,value):
        if Path(path).name=='FINAL_SEVERSTAL_BLIND_PROTOCOL_DRAFT.json':return
        save(path,value)
    ep.save=save_eval
    return ep,et,e
