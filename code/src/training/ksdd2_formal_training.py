"""Fresh Phase15C execution with stage-bound last.ckpt recovery."""
import csv
import gc
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from src.training import ksdd2_pilot_training as old
from src.evaluation import ksdd2_70k_protocol as p
from src.evaluation.ksdd2_training_calibration import components

def fresh(name):
    return old.fresh(name,tuple(p.config(name)['input_hw']))

def configure(m):
    # Official optimizer only queries this loader's length. The manual loop below
    # performs image loading using the same frozen1876 source rows and sequence.
    loader=torch.utils.data.DataLoader(p.base.assets()['groups']['train_normal'],batch_size=1,shuffle=False)
    m._trainer=SimpleNamespace(max_epochs=1000,max_steps=p.STEPS,datamodule=SimpleNamespace(train_dataloader=lambda:loader))
    c=m.configure_optimizers();o,s=c['optimizer'],c['lr_scheduler']
    if s.step_size!=66500 or s.gamma!=.1:raise ValueError('FAIL_SCHEDULER')
    return o,s

def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

def resume_payload(m,opt,sch,name,step,**extra):
    return dict(stage='PHASE15C',method=name,input_hw=p.config(name)['input_hw'],config=p.config(name),
        global_step=step,parent_sha=p.base.PARENT_SHA,split_sha=p.base.SPLIT_SHA,
        model_state=m.state_dict(),optimizer=opt.state_dict(),scheduler=sch.state_dict(),rng=rng_state(),**extra)

def restore(m,opt,sch,c,name):
    p.checkpoint_binding(c,name);m.load_state_dict(c['model_state'],strict=True)
    opt.load_state_dict(c['optimizer']);sch.load_state_dict(c['scheduler'])
    assert_exact_state(m.state_dict(),c['model_state'])
    assert_exact_state(opt.state_dict(),c['optimizer'])
    assert_exact_state(sch.state_dict(),c['scheduler'])
    if sch.last_epoch!=c['global_step']:raise ValueError('FAIL_SCHEDULER_STEP')
    random.setstate(c['rng']['python']);np.random.set_state(c['rng']['numpy']);torch.set_rng_state(c['rng']['torch'].cpu())
    if c['rng']['cuda']:torch.cuda.set_rng_state_all([x.cpu() for x in c['rng']['cuda']])
    return c['global_step']

def assert_exact_state(actual,expected):
    if isinstance(expected,torch.Tensor):
        if not torch.equal(actual.detach().cpu(),expected.detach().cpu()):raise ValueError('FAIL_EXACT_RESTORE')
    elif isinstance(expected,dict):
        if actual.keys()!=expected.keys():raise ValueError('FAIL_EXACT_RESTORE')
        for k in expected:assert_exact_state(actual[k],expected[k])
    elif isinstance(expected,(list,tuple)):
        if len(actual)!=len(expected):raise ValueError('FAIL_EXACT_RESTORE')
        for a,b in zip(actual,expected):assert_exact_state(a,b)
    elif actual!=expected:raise ValueError('FAIL_EXACT_RESTORE')

def continuation_error(actual,expected):
    error=0.
    for k,ref in expected.items():
        value=actual[k].detach().cpu();ref=ref.detach().cpu()
        if value.is_floating_point():
            torch.testing.assert_close(value,ref,rtol=0,atol=torch.finfo(value.dtype).eps)
            error=max(error,float((value-ref).abs().max()))
        elif not torch.equal(value,ref):raise AssertionError('FAIL_CONTINUATION_STATE')
    return error

def freeze_json(path,value):
    if path.exists() and p.base.read(path)!=value:raise ValueError('FAIL_FROZEN_PROTOCOL_CHANGED '+str(path))
    if not path.exists():p.base.save(path,value)

def freeze_sequences(a):
    ids=p.sequence([r['id'] for r in a['groups']['train_normal']]);path=p.OUT/'train_sequence_manifest.csv'
    expected=[dict(global_step=str(i+1),source_id=s) for i,s in enumerate(ids)]
    if path.exists():
        with path.open(encoding='utf-8') as f:
            if list(csv.DictReader(f))!=expected:raise ValueError('FAIL_TRAIN_SEQUENCE_CHANGED')
    else:p.base.csv_save(path,expected)
    return ids

def setup(m,name):
    hw=tuple(p.config(name)['input_hw']);stats_path=p.base.parent.OUT/f'teacher_stats_{hw[0]}x{hw[1]}.pt'
    frozen=p.base.read(p.base.OUT/'teacher_stats_binding.json')[stats_path.name]
    if p.base.parent.digest(stats_path)!=frozen:raise ValueError('FAIL_TEACHER_STATS_CHANGED')
    stats=torch.load(stats_path,map_location='cpu',weights_only=True);old.validate_teacher_stats(stats)
    m.model.mean_std.update({k:v.cuda() for k,v in stats.items()})
    old.cal.prepare_extra(m,hw);ds=m.imagenet_loader.dataset;order=p.sequence(list(range(len(ds))))
    paths=[str(Path(ds.samples[i][0]).resolve()) for i in order]
    path=p.OUT/'imagenette_sequence_manifest.csv';expected=[dict(global_step=str(i+1),source_path=s) for i,s in enumerate(paths)]
    if path.exists():
        with path.open(encoding='utf-8') as f:
            if list(csv.DictReader(f))!=expected:raise ValueError('FAIL_IMAGENETTE_SEQUENCE_CHANGED')
    else:p.base.csv_save(path,expected)
    return ds,order,paths

def verify_implementation(binding):
    if any(p.base.parent.digest(p.base.ROOT/file)!=sha for file,sha in binding.items()):raise ValueError('FAIL_IMPLEMENTATION_CHANGED')
    return True

def bindings(a):
    implementation=p.base.read(p.OUT/'implementation_binding.json');verify_implementation(implementation)
    return dict(train_sequence_sha=p.base.parent.digest(p.OUT/'train_sequence_manifest.csv'),
        imagenette_sequence_sha=p.base.parent.digest(p.OUT/'imagenette_sequence_manifest.csv'),
        pilot_report_sha=a['pilot_report_sha'],implementation=implementation)

def ensure_initial_checkpoint(path,payload):
    if not path.exists():old.atomic_save(path,payload)

def load_final(path,name):
    path=Path(path)
    if path.resolve()!=(p.OUT/p.FOLDERS[name]/'final.ckpt').resolve():raise ValueError('FAIL_PILOT_CONTAMINATION')
    c=torch.load(path,map_location='cpu',weights_only=False);p.final_gate(c,name)
    m=fresh(name);m.load_state_dict(c['model_state'],strict=True);m.eval()
    if old.state_hash(m)!=c['state_hash'] or old.quantile_values(m,name)!=c['quantiles']:raise ValueError('FAIL_STRICT_QUANTILE_RELOAD')
    return m,c

def save_report(path,c):
    r=dict(c['training_summary'],checkpoint=str(path.resolve()),checkpoint_size=path.stat().st_size,
        checkpoint_sha256=p.base.parent.digest(path),strict_reload=True,quantiles_reload=True,status='PASS')
    p.base.save(path.parent/'training_report.json',r)
    (path.parent/'training_report.txt').write_text(json.dumps(r,indent=2),encoding='utf-8')
    return r

def train_one(name,a,guard,seq,expected_init):
    cfg=p.config(name);folder=p.OUT/p.FOLDERS[name]
    for d in ('checkpoints','logs'):(folder/d).mkdir(parents=True,exist_ok=True)
    freeze_json(folder/'config.json',cfg);last=folder/'checkpoints/last.ckpt';final=folder/'final.ckpt'
    if final.exists():
        m,c=load_final(final,name)
        if c['bindings']!=bindings(a):raise ValueError('FAIL_FINAL_BINDINGS')
        r=save_report(final,c);del m,c;gc.collect();torch.cuda.empty_cache();return r
    m=fresh(name);initial=old.branches(m)
    if initial!=expected_init:raise ValueError('FAIL_FRESH_HASHES')
    ds,extra_seq,extra_paths=setup(m,name);old.cal.seed();opt,sch=configure(m)
    bound=bindings(a);logs=[];elapsed_before=0.;peak_a=peak_r=0.;resume_steps=[];step=0
    if last.exists():
        c=torch.load(last,map_location='cpu',weights_only=False);p.checkpoint_binding(c,name)
        if c.get('bindings')!=bound or c.get('fresh_hashes')!=initial:raise ValueError('FAIL_RESUME_BINDINGS')
        step=restore(m,opt,sch,c,name);logs=c['history'];elapsed_before=c['training_seconds']
        peak_a=c['peak_allocated_mib'];peak_r=c['peak_reserved_mib'];resume_steps=c['resume_from_steps']+[step]
        if len(logs)!=step or any(r['global_step']!=i+1 or r['source_id']!=seq[i] or r['imagenette_source']!=extra_paths[i] for i,r in enumerate(logs)):
            raise ValueError('FAIL_RESUME_HISTORY')
        del c;print(f'{name}: resume own Phase15C last.ckpt at {step}; optimizer/scheduler/RNG restored',flush=True)
    elif (folder/'loss_history.csv').exists():raise ValueError('FAIL_HISTORY_WITHOUT_CHECKPOINT')
    m.train();m.model.teacher.eval();lookup={r['id']:r for r in a['groups']['train_normal']};hw=tuple(cfg['input_hw'])
    ensure_initial_checkpoint(last,resume_payload(m,opt,sch,name,0,history=[],fresh_hashes=initial,bindings=bound,
        training_seconds=0.,peak_allocated_mib=0.,peak_reserved_mib=0.,resume_from_steps=[]))
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    with (folder/'loss_history.csv').open('w',newline='',encoding='utf-8') as f:
        writer=None
        if logs:
            writer=csv.DictWriter(f,fieldnames=list(logs[0]));writer.writeheader();writer.writerows(logs);f.flush()
        for i in range(step,p.STEPS):
            torch.cuda.synchronize();begin=time.perf_counter();sid=seq[i]
            x=old.normal_input(m,lookup[sid]['image_path'],guard,'train');extra=old.extra_input(ds,extra_seq[i]);opt.zero_grad(set_to_none=True)
            total,aux,row=components(m,x,extra,hw,False)
            if aux is not None or not torch.isfinite(total):raise FloatingPointError('TRAINING_CRITICAL_FAIL_LOSS')
            total.backward()
            if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in m.parameters()):raise FloatingPointError('TRAINING_CRITICAL_FAIL_GRADIENT')
            opt.step();sch.step();torch.cuda.synchronize()
            row.update(global_step=i+1,source_id=sid,imagenette_source=extra_paths[i],total_loss=float(total.detach()),
                lr=opt.param_groups[0]['lr'],seconds=time.perf_counter()-begin)
            logs.append(row)
            if writer is None:writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader()
            writer.writerow(row);del total,aux,x,extra
            if (i+1)%100==0:
                f.flush();print(f'{name} {i+1}/70000 | sec/step={np.mean([r["seconds"] for r in logs[-100:]]):.4f} | loss={row["total_loss"]:.6f} | lr={row["lr"]}',flush=True)
            if (i+1)%500==0:
                peak_a=max(peak_a,torch.cuda.max_memory_allocated()/1024**2);peak_r=max(peak_r,torch.cuda.max_memory_reserved()/1024**2)
                old.atomic_save(last,resume_payload(m,opt,sch,name,i+1,history=logs,fresh_hashes=initial,bindings=bound,
                    training_seconds=elapsed_before+time.perf_counter()-start,peak_allocated_mib=peak_a,peak_reserved_mib=peak_r,resume_from_steps=resume_steps))
    elapsed=elapsed_before+time.perf_counter()-start;after=old.branches(m)
    if initial['teacher']!=after['teacher'] or any(initial[k]==after[k] for k in ('student','ae')):raise ValueError('FAIL_PARAMETER_UPDATES')
    opt.zero_grad(set_to_none=True);q=old.validation(m,name,a,guard)
    summary=dict(**cfg,status='PASS',completed=p.STEPS,quantiles=q,quantiles_frozen=True,validation_count=209,
        training_hours=elapsed/3600,seconds_per_step=float(np.mean([r['seconds'] for r in logs[50:]])),
        peak_allocated_mib=peak_a,peak_reserved_mib=peak_r,fresh_hashes=initial,final_hashes=after,
        teacher_unchanged=True,student_updated=True,ae_updated=True,NaN=False,Inf=False,OOM=False,
        resume_used=bool(resume_steps),resume_from_steps=resume_steps,resume_from_step=resume_steps[-1] if resume_steps else 0,
        loss_final50={k:float(np.median([r[k] for r in logs[-50:]])) for k in ('st','ae','stae','official','total_loss')},
        bindings=bound,scheduler_final=sch.state_dict())
    c=resume_payload(m,opt,sch,name,p.STEPS,validation_count=209,quantiles=q,state_hash=old.state_hash(m),
        fresh_hashes=initial,bindings=bound,training_summary=summary)
    p.final_gate(c,name);old.atomic_save(final,c);restored,loaded=load_final(final,name)
    if old.state_hash(restored)!=old.state_hash(m):raise ValueError('FAIL_FINAL_RELOAD')
    report=save_report(final,c);print(f'{name}: 70000 +209 normal validation +strict quantile reload PASS; SHA={report["checkpoint_sha256"]}',flush=True)
    del m,opt,sch,ds,c,restored,loaded;gc.collect();torch.cuda.empty_cache();return report
