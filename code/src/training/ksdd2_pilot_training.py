"""Bounded10k pilot orchestration around unchanged Phase15A training helpers."""
import copy
import gc
import json
import random
import time
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
from src.evaluation import ksdd2_pilot_protocol as p
from src.evaluation import ksdd2_training_calibration as cal
from src.evaluation.efficientad_branch_audit import parameter_hash,state_hash
from src.evaluation.gradnorm_s1_calibration import preserved_rng
from src.evaluation.gradient_interference_audit import cosine
from src.models import multiscale_st_s1 as s1
from src.training import dynamic_gradnorm_s1 as dynamic

def fresh(name,hw):
    m=cal.fresh(hw)
    if name=='DGNB':
        for k in ('qa_s1','qb_s1'):m.register_buffer(k,torch.tensor(0.,device='cuda'))
    return m

def branches(m):return {n:parameter_hash(getattr(m.model,n)) for n in ('teacher','student','ae')}

def validate_teacher_stats(stats):
    if set(stats)!={'mean','std'} or any(v.shape!=(1,384,1,1) or not torch.isfinite(v).all() for v in stats.values()) or not (stats['std']>0).all():
        raise ValueError('FAIL_TEACHER_STATS')
    return True

def configure(m):
    m._trainer=SimpleNamespace(max_epochs=-1,max_steps=70000)
    c=m.configure_optimizers()
    if c['lr_scheduler'].step_size!=66500:raise ValueError('FAIL_SCHEDULER')
    return c['optimizer'],c['lr_scheduler']

def normal_input(m,path,guard,purpose):
    guard.check(path,purpose)
    return cal.image(m,path)

def extra_setup(m,hw):
    cal.prepare_extra(m,hw)
    ds=m.imagenet_loader.dataset
    seq=p.sequence(list(range(len(ds))))
    return ds,seq

def extra_input(ds,index):return ds[index][0][None].cuda()

def freeze_sequences(a):
    ids=[r['id'] for r in a['groups']['train_normal']];seq=p.sequence(ids)
    rows=[dict(global_step=i+1,source_id=sid) for i,sid in enumerate(seq)]
    f=p.OUT/'train_sequence_manifest.csv'
    if f.exists():
        import csv
        old=list(csv.DictReader(f.open(encoding='utf-8')))
        if [(int(v['global_step']),v['source_id']) for v in old]!=[(v['global_step'],v['source_id']) for v in rows]:raise ValueError('FAIL_SEQUENCE_CHANGED')
    else:p.csv_save(f,rows)
    return seq

def checkpoint_payload(m,opt,sch,controller,name,step,**kwargs):
    return dict(**p.FLAGS,method=name,global_step=step,model_state=m.state_dict(),
        input_hw=[256,256] if name=='STD' else [864,320],
        optimizer=opt.state_dict(),scheduler=sch.state_dict(),controller=controller.state_dict() if controller else None,
        parent_sha=p.PARENT_SHA,split_sha=p.SPLIT_SHA,rng=dict(python=random.getstate(),numpy=np.random.get_state(),
        torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()),**kwargs)

def atomic_save(path,payload):
    tmp=path.with_suffix(path.suffix+'.tmp');torch.save(payload,tmp);tmp.replace(path)

def quantile_values(m,name):
    q={k:float(v.detach()) for k,v in m.model.quantiles.items()}
    if name=='DGNB':q.update(qa_s1=float(m.qa_s1),qb_s1=float(m.qb_s1))
    return q

@torch.no_grad()
def validation(m,name,a,guard):
    m.eval();rows=a['groups']['validation_normal']
    class Batches:
        def __len__(self):return len(rows)
        def __iter__(self):
            for r in rows:yield SimpleNamespace(image=normal_input(m,r['image_path'],guard,'quantile'),gt_label=torch.tensor([0]))
    # Direct official normal-only calibration, including its >2**24 subsampling.
    m.model.quantiles.update(m.map_norm_quantiles(Batches()))
    if name=='DGNB':
        maps=[]
        for r in rows:
            x=normal_input(m,r['image_path'],guard,'quantile')
            with s1.Capture(m,s1.WANTED) as cap:m.model.compute_student_teacher_distance(x)
            maps.append(s1.s1_map(s1.discrepancy(cap.values['teacher',s1.S1],cap.values['student',s1.S1])).cpu())
        qa,qb=m._get_quantiles_of_maps(maps);m.qa_s1.copy_(qa);m.qb_s1.copy_(qb)
    return quantile_values(m,name)

def load_final(path,hw):
    c=torch.load(path,map_location='cpu',weights_only=False);p.checkpoint_gate(c)
    p.checkpoint_binding(c,c['method'],hw)
    if c['parent_sha']!=p.PARENT_SHA or c['split_sha']!=p.SPLIT_SHA:raise ValueError('FAIL_CHECKPOINT_BINDING')
    m=fresh(c['method'],hw);m.load_state_dict(c['model_state'],strict=True);m.eval()
    if state_hash(m)!=c['state_hash'] or quantile_values(m,c['method'])!=c['quantiles']:raise ValueError('FAIL_STRICT_RELOAD')
    if c['method']=='DGNB':
        ctrl=dynamic.Controller(1.);ctrl.load_state_dict(c['controller'])
        if ctrl.state_dict()!=c['controller'] or ctrl.steps!=10000:raise ValueError('FAIL_CONTROLLER_RELOAD')
    return m,c

def snapshot(m,ctrl,a,guard,ds,extra_seq,step,hw):
    before=state_hash(m);control=copy.deepcopy(ctrl.state_dict());rows=[]
    with preserved_rng():
        cal.seed();m.zero_grad(set_to_none=True)
        for i,r in enumerate(a['groups']['train_normal'][:16]):
            x=normal_input(m,r['image_path'],guard,'probe');extra=extra_input(ds,extra_seq[i])
            official,aux,_=cal.components(m,x,extra,hw,True)
            _,shared=dynamic.shared_parameters(m.model.student);no,ns,go,gs=dynamic.gradient_norms(official,aux,shared)
            lam=control['lambda_used'];rc=cosine(go+lam*gs,go)
            rows.append(dict(source_id=r['id'],weighted_ratio=lam*float(ns/no),cosine=cosine(go,gs),
                resultant_cosine=rc,resultant_angle=float(np.degrees(np.arccos(np.clip(rc,-1,1))))))
            del official,aux,x,extra,go,gs
        m.zero_grad(set_to_none=True)
    if state_hash(m)!=before or ctrl.state_dict()!=control:raise ValueError('FAIL_PROBE_MUTATION')
    return dict(step=step,lambda_used=control['lambda_used'],rows=rows,
        medians={k:float(np.median([r[k] for r in rows])) for k in ('weighted_ratio','cosine','resultant_cosine','resultant_angle')},
        strong_conflicts=sum(r['cosine']<-.25 for r in rows),state_hash_before=before,state_hash_after=state_hash(m),
        optimizer_steps=0,controller_unchanged=True,rng_restored=True)

def distribution(v):
    x=np.asarray(v,float)
    return dict(min=float(x.min()),max=float(x.max()),mean=float(x.mean()),median=float(np.median(x)),
        **{f'p{q}':float(np.percentile(x,q)) for q in (5,25,75,95)})

def train_one(name,a,guard,seq,expected_init):
    folder=p.OUT/p.FOLDERS[name];hw=tuple(a['parent']['methods'][name]['input_hw'])
    for sub in ('checkpoints','logs'):(folder/sub).mkdir(parents=True,exist_ok=True)
    if (folder/'final_10k.ckpt').exists() or (folder/'checkpoints/last.ckpt').exists():
        raise FileExistsError('FAIL_EXISTING_PILOT_USE_EXPLICIT_RECOVERY')
    config=dict(**p.FLAGS,method=name,**a['parent']['methods'][name],pilot_steps=10000)
    p.save(folder/'config.json',config)
    m=fresh(name,hw);initial=branches(m)
    if initial!=expected_init:raise ValueError('FAIL_FRESH_INITIALIZATION_MISMATCH')
    stats_path=p.parent.OUT/f'teacher_stats_{hw[0]}x{hw[1]}.pt'
    expected_stats=p.read(p.OUT/'teacher_stats_binding.json')
    if p.parent.digest(stats_path)!=expected_stats[stats_path.name]:raise ValueError('FAIL_TEACHER_STATS_CHANGED')
    stats=torch.load(stats_path,map_location='cuda',weights_only=True);validate_teacher_stats(stats);m.model.mean_std.update(stats)
    ds,extra_seq=extra_setup(m,hw)
    extra_paths=[str(Path(ds.samples[i][0]).resolve()) for i in extra_seq]
    sequence_file=p.OUT/'imagenette_sequence_manifest.csv'
    if sequence_file.exists():
        import csv
        if [r['source_path'] for r in csv.DictReader(sequence_file.open())]!=extra_paths:raise ValueError('FAIL_IMAGENETTE_SEQUENCE')
    else:p.csv_save(sequence_file,[dict(global_step=i+1,source_path=path) for i,path in enumerate(extra_paths)])
    # Fresh RNG reset after setup, same target/source ordering and augmentation streams.
    cal.seed();opt,sch=configure(m);ctrl=dynamic.Controller(1.) if name=='DGNB' else None
    m.train();m.model.teacher.eval();lookup={r['id']:r for r in a['groups']['train_normal']}
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();logs=[];probes={};shared=None
    import csv
    with (folder/'loss_history.csv').open('w',newline='',encoding='utf-8') as f:
        writer=None
        for i,sid in enumerate(seq):
            torch.cuda.synchronize();t=time.perf_counter();x=normal_input(m,lookup[sid]['image_path'],guard,'train');extra=extra_input(ds,extra_seq[i]);opt.zero_grad(set_to_none=True)
            official,aux,row=cal.components(m,x,extra,hw,ctrl is not None);total=official
            if ctrl:
                if shared is None:shared=dynamic.confirm_shared(m.model.student,official,aux)
                _,params=dynamic.shared_parameters(m.model.student);no,ns,_,_=dynamic.gradient_norms(official,aux,params)
                vals=ctrl.update(no,ns,split='train',normal=True);total=dynamic.total_loss(official,aux,vals['lambda_used'])
                row.update(s1_raw=float(aux.detach()),weighted_s1=float(aux.detach())*vals['lambda_used'],**vals)
            if not torch.isfinite(total):raise FloatingPointError('TRAINING_CRITICAL_FAIL_NONFINITE_LOSS')
            total.backward()
            if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in m.parameters()):raise FloatingPointError('TRAINING_CRITICAL_FAIL_NONFINITE_GRADIENT')
            opt.step();sch.step();torch.cuda.synchronize()
            row.update(global_step=i+1,source_id=sid,imagenette_source=extra_paths[i],total_loss=float(total.detach()),
                aux_fraction=row.get('weighted_s1',0.)/float(total.detach()),lr=opt.param_groups[0]['lr'],seconds=time.perf_counter()-t)
            logs.append(row)
            if writer is None:writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader()
            writer.writerow(row)
            del total,official,aux,x,extra
            if (i+1)%50==0:
                f.flush();print(f'{name} {i+1}/10000 | sec/step={np.mean([r["seconds"] for r in logs[-50:]]):.4f} | loss={row["total_loss"]:.6f}'+(f' | lambda={row["lambda_used"]:.6f}' if ctrl else ''),flush=True)
            if ctrl and i+1 in (200,1000,5000,10000):
                probes[str(i+1)]=snapshot(m,ctrl,a,guard,ds,extra_seq,i+1,hw);p.save(folder/'gradient_probe_report.json',dict(**p.FLAGS,probes=probes))
            if (i+1)%500==0:atomic_save(folder/'checkpoints/last.ckpt',checkpoint_payload(m,opt,sch,ctrl,name,i+1,fresh_hashes=initial))
    elapsed=time.perf_counter()-start;allocated=torch.cuda.max_memory_allocated()/1024**2;reserved=torch.cuda.max_memory_reserved()/1024**2
    finalbranches=branches(m)
    if initial['teacher']!=finalbranches['teacher'] or any(initial[k]==finalbranches[k] for k in ('student','ae')):raise ValueError('FAIL_PARAMETER_UPDATES')
    opt.zero_grad(set_to_none=True);q=validation(m,name,a,guard)
    c=checkpoint_payload(m,opt,sch,ctrl,name,10000,validation_count=209,quantiles=q,state_hash=state_hash(m),fresh_hashes=initial)
    p.checkpoint_gate(c);final=folder/'final_10k.ckpt';atomic_save(final,c)
    # Strict reload on a fresh instance with identical registered state, not partial loading.
    restored,loaded=load_final(final,hw)
    if state_hash(restored)!=state_hash(m):raise ValueError('FAIL_FINAL_STATE_RELOAD')
    report=dict(**p.FLAGS,status='PASS',completed=10000,strict_reload=True,quantiles_frozen=True,quantiles=q,
        training_hours=elapsed/3600,seconds_per_step=float(np.mean([r['seconds'] for r in logs[50:]])),
        peak_allocated_mib=allocated,peak_reserved_mib=reserved,fresh_hashes=initial,final_hashes=finalbranches,
        teacher_unchanged=True,student_updated=True,ae_updated=True,NaN=False,Inf=False,OOM=False,parameters=8058628,
        checkpoint=str(final),checkpoint_size=final.stat().st_size,checkpoint_sha256=p.parent.digest(final),
        shared=shared,controller_reload=True if ctrl else None,teacher_stats_source=str(stats_path),teacher_stats_sha256=p.parent.digest(stats_path),
        train_sequence_sha256=p.parent.digest(p.OUT/'train_sequence_manifest.csv'),imagenette_sequence_sha256=p.parent.digest(sequence_file),
        loss_final50={k:float(np.median([r[k] for r in logs[-50:]])) for k in ('st','ae','stae','official','total_loss')})
    if ctrl:
        report.update(controller=ctrl.state_dict(),lambda_stats=distribution([r['lambda_used'] for r in logs]),
            boundary_hit_count=ctrl.lower_boundary_hits+ctrl.upper_boundary_hits,boundary_hit_percent=100*(ctrl.lower_boundary_hits+ctrl.upper_boundary_hits)/10000,
            gradient_probes=probes)
    p.save(folder/'training_report.json',report);(folder/'training_report.txt').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'{name}: 10000/10000 +209 normal quantiles +strict reload PASS, SHA={report["checkpoint_sha256"]}',flush=True)
    del restored,loaded,c,m,opt,sch,ds;gc.collect();torch.cuda.empty_cache()
    return report
