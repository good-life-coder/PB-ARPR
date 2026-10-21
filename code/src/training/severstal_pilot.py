"""Severstal bounded fresh pilots; shared official losses and quantile routines."""
import csv
import gc
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from src.evaluation import severstal_protocol as p
from src.evaluation import ksdd2_training_calibration as cal
from src.evaluation.ksdd2_training_calibration import components
from src.training import ksdd2_pilot_training as old

def sequence(ids,steps):
    order=np.random.default_rng(42).permutation(len(ids))
    return [ids[int(order[i%len(ids)])] for i in range(steps)]
def check_steps(steps):
    if steps not in (3,200,10000):raise ValueError('FAIL_BOUNDED_STEPS')
    return True

def validate_ae_geometry(encoder,hw):
    shape=list(hw)
    for name,layer in encoder.named_modules():
        if isinstance(layer,torch.nn.Conv2d):
            out=[(shape[i]+2*layer.padding[i]-layer.dilation[i]*(layer.kernel_size[i]-1)-1)//layer.stride[i]+1 for i in (0,1)]
            if min(out)<=0:raise ValueError(f'FAIL_RECT_AE_GEOMETRY: {name} input={shape}, kernel={list(layer.kernel_size)}, output={out}; frozen geometry not modified')
            shape=out
    return shape
def all_ready(reports):
    if set(reports)!={'STD','RECT'} or any(r.get('status')!='PASS' or r.get('completed')!=10000 or not r.get('strict_reload') for r in reports.values()):raise ValueError('FAIL_BOTH_PILOTS_GATE')
    return True
def folder(name,a):return p.PILOT/('std_256_seed42' if name=='STD' else 'rect_'+'x'.join(map(str,a['geometry']['rect_hw']))+'_seed42')
def checkpoint_gate(c,name,hw,nv):
    if c['global_step']!=10000 or c['method']!=name or list(hw)!=c['input_hw'] or c['validation_count']!=nv:raise ValueError('FAIL_CHECKPOINT_PROTOCOL')
    q=c['quantiles']
    if set(q)!={'qa_st','qb_st','qa_ae','qb_ae'} or not all(np.isfinite(v) for v in q.values()) or any(q['qa_'+k]>=q['qb_'+k] for k in ('st','ae')):raise ValueError('FAIL_QUANTILES')
    return True
def load_final(name,a):
    report=p.read(folder(name,a)/'training_report.json');path=folder(name,a)/'final_10k.ckpt'
    if p.digest(path)!=report['checkpoint_sha256']:raise ValueError('FAIL_CHECKPOINT_SHA')
    c=torch.load(path,map_location='cpu',weights_only=False);hw=a['methods'][name]['input_hw'];checkpoint_gate(c,name,hw,len(a['groups']['validation_normal']))
    if c['split_sha']!=a['protocol']['canonical_split_sha256'] or c['geometry_sha']!=a['protocol']['geometry_sha256']:raise ValueError('FAIL_CHECKPOINT_BINDING')
    m=cal.fresh(hw);m.load_state_dict(c['model_state'],strict=True);m.eval()
    if old.state_hash(m)!=c['state_hash'] or old.quantile_values(m,name)!=c['quantiles']:raise ValueError('FAIL_STRICT_RELOAD')
    return m,c

def input_image(m,r,guard,purpose):
    guard.check(r['image_path'],purpose)
    return cal.image(m,r['image_path'])

def setup(a,guard):
    expected=None;shapes={}
    for name in ('STD','RECT'):
        hw=a['methods'][name]['input_hw'];m=cal.fresh(hw);initial=old.branches(m)
        try:validate_ae_geometry(m.model.ae.encoder,hw)
        except ValueError as exc:
            p.save(p.OUT/'ae_geometry_gate.json',dict(status='FAIL_RECT_AE_GEOMETRY',method=name,input_hw=hw,error=str(exc),geometry_changed=False,AE_changed=False))
            raise
        if expected is None:expected=initial
        if initial!=expected:raise ValueError('FAIL_FRESH_GEOMETRY_FAIRNESS')
        first=a['groups']['train_normal'][0];guard.check(first['image_path'],'probe');shapes[name]=cal.shape_probe(m,first['image_path'],hw)
        dest=p.OUT/f'teacher_stats_{name}.pt'
        if not dest.exists():
            paths=[]
            for row in a['groups']['train_normal']:guard.check(row['image_path'],'train');paths.append(row['image_path'])
            stats=cal.teacher_stats(m,paths);torch.save(stats,dest)
        else:stats=torch.load(dest,weights_only=True,map_location='cuda')
        old.validate_teacher_stats(stats)
        del m,stats;gc.collect();torch.cuda.empty_cache()
    p.save(p.OUT/'model_precheck.json',dict(status='PASS',shapes=shapes,fresh_hashes=expected,teacher_stats_sha={n:p.digest(p.OUT/f'teacher_stats_{n}.pt') for n in ('STD','RECT')}))
    return expected

def precheck():
    a=p.assets();p.geometry_regression();guard=p.Guard(a['rows']);guard.install()
    # Rehash only allowed train/validation/dev assets; blind source files stay locked.
    for role in ('train_normal','validation_normal','dev_anomaly'):
        for row in a['groups'][role]:
            if p.digest(row['image_path'])!=row['image_sha']:raise ValueError('FAIL_SOURCE_SHA')
            p.verify_mask(row)
    expected=setup(a,guard)
    p.PILOT.mkdir(parents=True,exist_ok=True)
    p.save(p.PILOT/'config.json',dict(methods=a['methods'],split_sha=a['protocol']['canonical_split_sha256'],geometry_sha=a['protocol']['geometry_sha256'],**p.FLAGS))
    print('PRECHECK PASS | CSV/RLE/split/geometry PASS | KSDD2 864x320 PASS | final blind LOCKED',flush=True)
    return a,guard,expected

def train(name,steps,a,guard,expected):
    check_steps(steps);hw=tuple(a['methods'][name]['input_hw']);dest=folder(name,a) if steps==10000 else p.OUT/f'{name}_{steps}'
    dest.mkdir(parents=True,exist_ok=True);(dest/'checkpoints').mkdir(exist_ok=True)
    reportpath=dest/'training_report.json'
    if reportpath.exists():
        report=p.read(reportpath)
        if report['status']!='PASS':raise ValueError('FAIL_PREVIOUS_RUN')
        if steps==10000:
            m,c=load_final(name,a);del m,c;gc.collect();torch.cuda.empty_cache()
        return report
    if (dest/'loss_history.csv').exists():raise ValueError('FAIL_INTERRUPTED_RUN_REQUIRES_EXACT_RECOVERY')
    m=cal.fresh(hw);initial=old.branches(m)
    if initial!=expected:raise ValueError('FAIL_FRESH_INITIALIZATION')
    statsfile=p.OUT/f'teacher_stats_{name}.pt'
    if p.digest(statsfile)!=p.read(p.OUT/'model_precheck.json')['teacher_stats_sha'][name]:raise ValueError('FAIL_TEACHER_STATS_SHA')
    m.model.mean_std.update(torch.load(statsfile,map_location='cuda',weights_only=True));cal.prepare_extra(m,hw);ds=m.imagenet_loader.dataset
    seq=sequence([r['id'] for r in a['groups']['train_normal']],steps);extra_seq=sequence(list(range(len(ds))),steps);extra_paths=[str(Path(ds.samples[i][0]).resolve()) for i in extra_seq]
    train_hash=__import__('hashlib').sha256(json.dumps(seq).encode()).hexdigest();extra_hash=__import__('hashlib').sha256(json.dumps(extra_paths).encode()).hexdigest()
    if steps==10000:
        for filename,rows in [('train_sequence_manifest.csv',[dict(global_step=i+1,source_id=s) for i,s in enumerate(seq)]),('imagenette_sequence_manifest.csv',[dict(global_step=i+1,source_path=s) for i,s in enumerate(extra_paths)])]:
            path=p.PILOT/filename
            if path.exists():
                previous=list(csv.DictReader(path.open(encoding='utf-8')))
                if [{k:str(v) for k,v in row.items()} for row in rows]!=previous:raise ValueError('FAIL_SEQUENCE_FAIRNESS')
            else:p.csv_save(path,rows)
    cal.seed();opt,sch=old.configure(m);m.train();m.model.teacher.eval();lookup={r['id']:r for r in a['groups']['train_normal']}
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();logs=[]
    with (dest/'loss_history.csv').open('w',newline='',encoding='utf-8') as f:
        writer=None
        for i,sid in enumerate(seq):
            torch.cuda.synchronize();t=time.perf_counter();x=input_image(m,lookup[sid],guard,'train');extra=old.extra_input(ds,extra_seq[i]);opt.zero_grad(set_to_none=True)
            loss,aux,row=components(m,x,extra,hw,False)
            if not torch.isfinite(loss):raise FloatingPointError('FAIL_NONFINITE_LOSS')
            loss.backward()
            if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in m.parameters()):raise FloatingPointError('FAIL_NONFINITE_GRADIENT')
            opt.step();sch.step();torch.cuda.synchronize();row.update(global_step=i+1,source_id=sid,lr=opt.param_groups[0]['lr'],seconds=time.perf_counter()-t)
            logs.append(row)
            if writer is None:writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader()
            writer.writerow(row);del x,extra,loss,aux
            if (i+1)%50==0:f.flush();print(f'{name} {i+1}/{steps} | sec/step={np.mean([v["seconds"] for v in logs[-50:]]):.4f}',flush=True)
            if steps==10000 and (i+1)%500==0:old.atomic_save(dest/'checkpoints/last.ckpt',dict(global_step=i+1,model_state=m.state_dict(),optimizer=opt.state_dict(),scheduler=sch.state_dict(),method=name,input_hw=list(hw),split_sha=a['protocol']['canonical_split_sha256'],geometry_sha=a['protocol']['geometry_sha256'],rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())))
    hours=(time.perf_counter()-start)/3600;allocated=torch.cuda.max_memory_allocated()/2**20;reserved=torch.cuda.max_memory_reserved()/2**20;end=old.branches(m)
    if initial['teacher']!=end['teacher'] or any(initial[k]==end[k] for k in ('student','ae')):raise ValueError('FAIL_BRANCH_UPDATES')
    seconds=float(np.mean([v['seconds'] for v in logs[50:] or logs]));report=dict(status='PASS',method=name,completed=steps,input_hw=list(hw),training_hours=hours,seconds_per_step=seconds,projected10k_hours=seconds*10000/3600,projected70k_hours=seconds*70000/3600,peak_allocated_mib=allocated,peak_reserved_mib=reserved,teacher_unchanged=True,student_updated=True,ae_updated=True,NaN=False,Inf=False,OOM=False,parameters=8058628,fresh_hashes=initial,final_hashes=end,training_sequence_hash=train_hash,ImageNette_sequence_hash=extra_hash,**p.FLAGS)
    if steps==10000:
        opt.zero_grad(set_to_none=True);q=old.validation(m,name,a,guard)
        c=dict(global_step=steps,method=name,input_hw=list(hw),model_state=m.state_dict(),quantiles=q,validation_count=len(a['groups']['validation_normal']),state_hash=old.state_hash(m),split_sha=a['protocol']['canonical_split_sha256'],geometry_sha=a['protocol']['geometry_sha256'],optimizer=opt.state_dict(),scheduler=sch.state_dict())
        checkpoint_gate(c,name,hw,len(a['groups']['validation_normal']));path=dest/'final_10k.ckpt';old.atomic_save(path,c)
        report.update(checkpoint=str(path),checkpoint_sha256=p.digest(path),checkpoint_size=path.stat().st_size,quantiles=q,strict_reload=False)
        p.save(reportpath,report);restored,loaded=load_final(name,a)
        if old.state_hash(restored)!=old.state_hash(m):raise ValueError('FAIL_RELOAD_STATE')
        report['strict_reload']=True;del restored,loaded,c
    p.save(reportpath,report);(dest/'training_report.txt').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'{name} {steps}/{steps}: PASS',flush=True)
    del m,opt,sch,ds;gc.collect();torch.cuda.empty_cache();return report
