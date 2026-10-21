"""Seed-aware orchestration reusing Phase15C official training primitives."""
import csv
import gc
import json
import random
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from lightning.pytorch import seed_everything
from anomalib.models import EfficientAd
from src.training import ksdd2_formal_training as core
from src.evaluation import ksdd2_repeat_summary as p

old=core.old
components=core.components
FILES=['src/training/ksdd2_repeat_training.py','src/evaluation/ksdd2_repeat_summary.py',
    'scripts/train_ksdd2_repeat_seeds.py','scripts/evaluate_ksdd2_repeat_dev.py']

def seed_rng(seed):seed_everything(seed,workers=True)
def now():return datetime.now(timezone.utc).isoformat()

def fresh(seed,name):
    cfg=p.config(seed,name);seed_rng(seed)
    # Same official constructor/pretrained loading as Phase15C; only seed varies.
    m=EfficientAd(model_size='small',imagenet_dir=str(p.base.ROOT/'datasets/imagenette'),
        pre_processor=EfficientAd.configure_pre_processor(image_size=tuple(cfg['input_hw'])),
        evaluator=False,post_processor=False,visualizer=False)
    m.prepare_pretrained_model();m.cuda();m.eval()
    if sum(v.numel() for v in m.parameters())!=8058628:raise ValueError('FAIL_PARAMETERS')
    for v in m.model.teacher.parameters():v.requires_grad_(False)
    return m

def references():
    results={}
    core.verify_implementation(p.base.read(p.parent.OUT/'implementation_binding.json'))
    for name in p.NAMES:
        path=p.parent.OUT/p.parent.FOLDERS[name]/'final.ckpt'
        if not path.is_file() or p.base.parent.digest(path)!=p.REFERENCE_SHA[name]:raise ValueError('FAIL_SEED42_REFERENCE')
        m,c=core.load_final(path,name);r=p.base.read(path.parent/'training_report.json')
        if not p.healthy(r) or r['checkpoint_sha256']!=p.REFERENCE_SHA[name] or r['quantiles']!=c['quantiles']:raise ValueError('FAIL_SEED42_REFERENCE')
        results[name]=r;del m,c;gc.collect();torch.cuda.empty_cache()
    return results

def payload(m,opt,sch,seed,name,step,**extra):
    return dict(stage='PHASE15D',seed=seed,method=name,config=p.config(seed,name),input_hw=p.config(seed,name)['input_hw'],
        global_step=step,parent_sha=p.base.PARENT_SHA,split_sha=p.base.SPLIT_SHA,model_state=m.state_dict(),
        optimizer=opt.state_dict(),scheduler=sch.state_dict(),rng=core.rng_state(),**extra)

def restore(m,opt,sch,c,seed,name):
    p.checkpoint_gate(c,seed,name);m.load_state_dict(c['model_state'],strict=True);opt.load_state_dict(c['optimizer']);sch.load_state_dict(c['scheduler'])
    for actual,expected in ((m.state_dict(),c['model_state']),(opt.state_dict(),c['optimizer']),(sch.state_dict(),c['scheduler'])):core.assert_exact_state(actual,expected)
    if sch.last_epoch!=c['global_step']:raise ValueError('FAIL_SCHEDULER_STEP')
    random.setstate(c['rng']['python']);np.random.set_state(c['rng']['numpy']);torch.set_rng_state(c['rng']['torch'].cpu())
    if c['rng']['cuda']:torch.cuda.set_rng_state_all([v.cpu() for v in c['rng']['cuda']])
    return c['global_step']

def freeze_csv(path,rows):
    expected=[{k:str(v) for k,v in r.items()} for r in rows]
    if path.exists():
        with path.open(encoding='utf-8') as f:
            if list(csv.DictReader(f))!=expected:raise ValueError('FAIL_SEQUENCE_CHANGED')
    else:p.base.csv_save(path,rows)

def sequence(a,seed):
    ids=p.sequence([r['id'] for r in a['groups']['train_normal']],seed)
    freeze_csv(p.OUT/f'seed{seed}'/'train_sequence_manifest.csv',[dict(global_step=i+1,source_id=s) for i,s in enumerate(ids)])
    return ids

def setup(m,seed,name):
    hw=tuple(p.config(seed,name)['input_hw']);path=p.base.parent.OUT/f'teacher_stats_{hw[0]}x{hw[1]}.pt'
    expected=p.base.read(p.base.OUT/'teacher_stats_binding.json')[path.name]
    if p.base.parent.digest(path)!=expected:raise ValueError('FAIL_TEACHER_STATS_CHANGED')
    stats=torch.load(path,map_location='cpu',weights_only=True);old.validate_teacher_stats(stats)
    m.model.mean_std.update({k:v.cuda() for k,v in stats.items()})
    old.cal.prepare_extra(m,hw);ds=m.imagenet_loader.dataset;order=p.sequence(list(range(len(ds))),seed)
    paths=[str(Path(ds.samples[i][0]).resolve()) for i in order]
    freeze_csv(p.OUT/f'seed{seed}'/'imagenette_sequence_manifest.csv',[dict(global_step=i+1,source_path=s) for i,s in enumerate(paths)])
    return ds,order,paths

def binding(seed):
    implementation=p.base.read(p.OUT/'implementation_binding.json');core.verify_implementation(implementation)
    return dict(implementation=implementation,training_sequence_hash=p.base.parent.digest(p.OUT/f'seed{seed}'/'train_sequence_manifest.csv'),
        ImageNette_sequence_hash=p.base.parent.digest(p.OUT/f'seed{seed}'/'imagenette_sequence_manifest.csv'),
        development_assets_sha=p.base.parent.digest(p.OUT/'development_asset_binding.json'))

def load_final(seed,name):
    path=p.folder(seed,name)/'final.ckpt';c=torch.load(path,map_location='cpu',weights_only=False)
    p.checkpoint_gate(c,seed,name,final=True)
    if c['bindings']!=binding(seed):raise ValueError('FAIL_FINAL_BINDINGS')
    m=fresh(seed,name);m.load_state_dict(c['model_state'],strict=True);m.eval()
    if old.state_hash(m)!=c['state_hash'] or old.quantile_values(m,name)!=c['quantiles']:raise ValueError('FAIL_STRICT_RELOAD')
    return m,c

def report(path,c):
    r=dict(c['training_summary'],checkpoint=str(path.resolve()),checkpoint_size=path.stat().st_size,
        checkpoint_sha256=p.base.parent.digest(path),strict_reload=True,quantiles_reload=True,status='PASS')
    p.base.save(path.parent/'training_report.json',r);(path.parent/'training_report.txt').write_text(json.dumps(r,indent=2),encoding='utf-8')
    return r

def train_one(seed,name,a,guard,expected):
    cfg=p.config(seed,name);folder=p.folder(seed,name);seq=sequence(a,seed)
    for d in ('logs','checkpoints'):(folder/d).mkdir(parents=True,exist_ok=True)
    core.freeze_json(folder/'config.json',cfg);final=folder/'final.ckpt';last=folder/'checkpoints/last.ckpt'
    if final.exists():
        prior=p.base.read(folder/'training_report.json') if (folder/'training_report.json').exists() else None
        if prior and prior['checkpoint_sha256']!=p.base.parent.digest(final):raise ValueError('FAIL_FINAL_SHA_CHANGED')
        m,c=load_final(seed,name);r=report(final,c);del m,c;gc.collect();torch.cuda.empty_cache();return r
    m=fresh(seed,name);initial=old.branches(m)
    if initial!=expected:raise ValueError('FAIL_FRESH_HASHES')
    ds,order,paths=setup(m,seed,name);seed_rng(seed);opt,sch=core.configure(m);bound=binding(seed)
    logs=[];step=0;elapsed_before=0.;peak_a=peak_r=0.;resumes=[];timestamp_start=now()
    if last.exists():
        c=torch.load(last,map_location='cpu',weights_only=False);p.checkpoint_gate(c,seed,name)
        if c.get('bindings')!=bound or c.get('fresh_hashes')!=initial:raise ValueError('FAIL_RESUME_BINDING')
        step=restore(m,opt,sch,c,seed,name);logs=c['history'];elapsed_before=c['training_seconds'];timestamp_start=c['timestamp_start']
        peak_a=c['peak_allocated_mib'];peak_r=c['peak_reserved_mib'];resumes=c['resume_from_steps']+[step]
        if len(logs)!=step or any(r['global_step']!=i+1 or r['source_id']!=seq[i] or r['imagenette_source']!=paths[i] for i,r in enumerate(logs)):raise ValueError('FAIL_RESUME_HISTORY')
        del c;print(f'Seed{seed} {name}: own last.ckpt resumed at {step}',flush=True)
    elif (folder/'loss_history.csv').exists():raise ValueError('FAIL_HISTORY_WITHOUT_LAST')
    def save_last(current,seconds):
        old.atomic_save(last,payload(m,opt,sch,seed,name,current,history=logs,bindings=bound,fresh_hashes=initial,
            timestamp_start=timestamp_start,training_seconds=seconds,peak_allocated_mib=peak_a,peak_reserved_mib=peak_r,resume_from_steps=resumes))
    if not last.exists():save_last(0,0.)
    m.train();m.model.teacher.eval();lookup={r['id']:r for r in a['groups']['train_normal']};hw=tuple(cfg['input_hw'])
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    # Disk history is reconciled to the last committed checkpoint on recovery.
    with (folder/'loss_history.csv').open('w',newline='',encoding='utf-8') as f:
        writer=None
        if logs:writer=csv.DictWriter(f,fieldnames=list(logs[0]));writer.writeheader();writer.writerows(logs);f.flush()
        for i in range(step,70000):
            torch.cuda.synchronize();begin=time.perf_counter();sid=seq[i]
            x=old.normal_input(m,lookup[sid]['image_path'],guard,'train');extra=old.extra_input(ds,order[i]);opt.zero_grad(set_to_none=True)
            total,aux,row=components(m,x,extra,hw,False)
            if aux is not None or not torch.isfinite(total):raise FloatingPointError('TRAINING_CRITICAL_FAIL_LOSS')
            total.backward()
            if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in m.parameters()):raise FloatingPointError('TRAINING_CRITICAL_FAIL_GRADIENT')
            opt.step();sch.step();torch.cuda.synchronize()
            row.update(global_step=i+1,source_id=sid,imagenette_source=paths[i],total_loss=float(total.detach()),lr=opt.param_groups[0]['lr'],seconds=time.perf_counter()-begin)
            logs.append(row)
            if writer is None:writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader()
            writer.writerow(row);del total,aux,x,extra
            if (i+1)%100==0:
                f.flush();print(f'Seed{seed} {name} {i+1}/70000 | sec/step={np.mean([r["seconds"] for r in logs[-100:]]):.4f} | loss={row["total_loss"]:.6f}',flush=True)
            if (i+1)%500==0:
                peak_a=max(peak_a,torch.cuda.max_memory_allocated()/1024**2);peak_r=max(peak_r,torch.cuda.max_memory_reserved()/1024**2)
                save_last(i+1,elapsed_before+time.perf_counter()-start)
    elapsed=elapsed_before+time.perf_counter()-start;training_end=now();after=old.branches(m)
    if initial['teacher']!=after['teacher'] or any(initial[k]==after[k] for k in ('student','ae')):raise ValueError('FAIL_PARAMETER_UPDATES')
    opt.zero_grad(set_to_none=True);q=old.validation(m,name,a,guard)
    summary=dict(**cfg,status='PASS',completed=70000,timestamp_start=timestamp_start,timestamp_training_end=training_end,timestamp_end=now(),
        training_hours=elapsed/3600,seconds_per_step=float(np.mean([r['seconds'] for r in logs[50:]])),peak_allocated_mib=peak_a,peak_reserved_mib=peak_r,
        fresh_hashes=initial,final_hashes=after,teacher_unchanged=True,student_updated=True,ae_updated=True,NaN=False,Inf=False,OOM=False,
        quantiles=q,quantiles_frozen=True,validation_count=209,resume_used=bool(resumes),resume_from_steps=resumes,resume_from_step=resumes[-1] if resumes else 0,
        bindings=bound,training_sequence_hash=bound['training_sequence_hash'],ImageNette_sequence_hash=bound['ImageNette_sequence_hash'],
        scheduler_final=sch.state_dict(),loss_final50={k:float(np.median([r[k] for r in logs[-50:]])) for k in ('st','ae','stae','official','total_loss')})
    c=payload(m,opt,sch,seed,name,70000,validation_count=209,quantiles=q,state_hash=old.state_hash(m),bindings=bound,fresh_hashes=initial,training_summary=summary)
    p.checkpoint_gate(c,seed,name,final=True);old.atomic_save(final,c);reloaded,loaded=load_final(seed,name)
    if old.state_hash(reloaded)!=old.state_hash(m):raise ValueError('FAIL_FINAL_RELOAD')
    r=report(final,c);print(f'Seed{seed} {name}: final70000 +209 normal validation +strict reload PASS; SHA={r["checkpoint_sha256"]}',flush=True)
    del m,opt,sch,ds,c,reloaded,loaded;gc.collect();torch.cuda.empty_cache();return r
