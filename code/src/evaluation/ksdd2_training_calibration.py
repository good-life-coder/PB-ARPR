"""Bounded external smoke/timing using unchanged official losses and DGNB controller."""
import gc
import random
import time
from types import SimpleNamespace
import numpy as np
import torch
from anomalib.models import EfficientAd
from anomalib.data.utils import read_image
from src.models import multiscale_st_s1 as s1
from src.training import dynamic_gradnorm_s1 as dynamic
from src.evaluation.efficientad_branch_audit import parameter_hash
from src.evaluation import ksdd2_external_protocol as protocol

def seed():random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)

def fresh(hw):
    seed();m=EfficientAd(model_size='small',imagenet_dir=str(protocol.ROOT/'datasets/imagenette'),pre_processor=EfficientAd.configure_pre_processor(image_size=tuple(hw)),evaluator=False,post_processor=False,visualizer=False)
    m.prepare_pretrained_model();m.cuda();m.eval()
    if sum(p.numel() for p in m.parameters())!=8058628:raise ValueError('FAIL_PARAMETERS')
    for p in m.model.teacher.parameters():p.requires_grad_(False)
    return m

@torch.no_grad()
def image(model,path):return model.pre_processor(read_image(path,as_tensor=True)[None].cuda())

@torch.no_grad()
def shape_probe(m,path,hw):
    x=image(m,path)
    with s1.Capture(m,s1.WANTED) as cap:student,distance=m.model.compute_student_teacher_distance(x)
    teacher=cap.values['teacher',s1.S1];student_s1=cap.values['student',s1.S1];expected=protocol.feature_shapes(hw)
    if list(teacher.shape[1:])!=expected['S1'] or teacher.shape!=student_s1.shape or list(distance.shape[1:])!=expected['Final']:raise ValueError('FAIL_NATURAL_MATCH')
    return dict(input=list(x.shape),teacher_s1=list(teacher.shape[1:]),student_s1=list(student_s1.shape[1:]),final_st=list(distance.shape[1:]),natural_match=True,projector=False,params=8058628)

def prepare_extra(m,hw):m.prepare_imagenette_data(tuple(hw))

def next_extra(m):
    try:x=next(m.imagenet_iterator)[0]
    except StopIteration:m.imagenet_iterator=iter(m.imagenet_loader);x=next(m.imagenet_iterator)[0]
    return x.cuda()

@torch.no_grad()
def teacher_stats(m,paths):
    class Batches:
        def __len__(self):return len(paths)
        def __iter__(self):
            for i,p in enumerate(paths):
                if i%400==0:print(f'Teacher normal stats {i}/{len(paths)}',flush=True)
                yield SimpleNamespace(image=image(m,p))
    # Same official full-train channel moments, no anomaly or validation samples.
    result=m.teacher_channel_mean_std(Batches())
    if not all(torch.isfinite(v).all() for v in result.values()) or not (result['std']>0).all():raise ValueError('FAIL_TEACHER_STATS')
    return {k:v.detach().clone() for k,v in result.items()}

def components(m,x,extra,hw,is_dgnb):
    if x.shape!=(1,3,*hw) or x.dtype!=torch.float32:raise ValueError('FAIL_INPUT')
    with s1.Capture(m,s1.WANTED if is_dgnb else {}) as capture:st,ae,stae=m.model(batch=x,batch_imagenet=extra)
    official=st+ae+stae;aux=None
    if is_dgnb:
        teacher=capture.values['teacher',s1.S1];student=capture.values['student',s1.S1]
        if teacher.shape!=student.shape or list(teacher.shape[1:])!=protocol.feature_shapes(hw)['S1']:raise ValueError('FAIL_S1')
        aux=s1.hard_loss(s1.discrepancy(teacher,student))
    return official,aux,dict(st=float(st.detach()),ae=float(ae.detach()),stae=float(stae.detach()),official=float(official.detach()))

def run(name,hw,paths,stats,steps):
    if steps not in [3,200]:raise ValueError('FAIL_BOUNDED_STEPS')
    m=fresh(hw);m.model.mean_std.update(stats);prepare_extra(m,hw);controller=dynamic.Controller(1.) if name=='DGNB' else None
    # Official configure_optimizers computes the frozen 70k scheduler; calibration does not compress it.
    m._trainer=SimpleNamespace(max_epochs=-1,max_steps=70000);configured=m.configure_optimizers();opt=configured['optimizer'];scheduler=configured['lr_scheduler'];m.train();m.model.teacher.eval()
    teacher_before=parameter_hash(m.model.teacher);student_before=parameter_hash(m.model.student);ae_before=parameter_hash(m.model.ae)
    sequence=np.random.default_rng(42).permutation(len(paths));logs=[];shared=None;torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    for i in range(steps):
        torch.cuda.synchronize();start=time.perf_counter();path=paths[int(sequence[i%len(sequence)])];x=image(m,path);extra=next_extra(m);opt.zero_grad(set_to_none=True)
        official,aux,row=components(m,x,extra,hw,controller is not None);total=official
        if controller is not None:
            if shared is None:shared=dynamic.confirm_shared(m.model.student,official,aux)
            _,params=dynamic.shared_parameters(m.model.student);off,naux,_,_=dynamic.gradient_norms(official,aux,params);values=controller.update(off,naux,split='train',normal=True);total=dynamic.total_loss(official,aux,values['lambda_used']);row.update(s1_raw=float(aux.detach()),**values)
        if not torch.isfinite(total):raise FloatingPointError('FAIL_NONFINITE_LOSS')
        total.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in m.parameters()):raise FloatingPointError('FAIL_NONFINITE_GRADIENT')
        opt.step();scheduler.step();torch.cuda.synchronize();row.update(step=i+1,path=str(path),total=float(total.detach()),seconds=time.perf_counter()-start,lr=opt.param_groups[0]['lr']);logs.append(row)
        if (i+1)%50==0:print(f'{name} {i+1}/{steps}, loss={row["total"]:.5f}, sec/step={np.mean([v["seconds"] for v in logs[-50:]]):.4f}',flush=True)
    if parameter_hash(m.model.teacher)!=teacher_before or parameter_hash(m.model.student)==student_before or parameter_hash(m.model.ae)==ae_before:raise ValueError('FAIL_UPDATE_INTEGRITY')
    times=[v['seconds'] for v in logs[50:] or logs];r=dict(status='PASS',completed=steps,seconds_per_step=float(np.mean(times)),projected70k_hours=float(np.mean(times))*70000/3600,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,teacher_unchanged=True,student_updated=True,ae_updated=True,NaN=False,Inf=False,OOM=False,params=8058628,shared=shared,steps_for_steady='51-200' if steps==200 else 'all3')
    if controller:
        state=controller.state_dict();restored=dynamic.Controller(1.);restored.load_state_dict(state)
        if state!=restored.state_dict():raise ValueError('FAIL_CONTROLLER_RELOAD')
        dest=protocol.OUT/f'{name}_{steps}_controller_state.pt';torch.save(dict(controller=state,optimizer=opt.state_dict(),step=steps),dest);loaded=torch.load(dest,weights_only=False);restored.load_state_dict(loaded['controller']);r.update(controller=state,controller_serialization=True)
    protocol.save(f'{name}_{steps}_report.json',r)
    import csv
    with (protocol.OUT/f'{name}_{steps}_training_log.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(logs[0]));w.writeheader();w.writerows(logs)
    del m,opt,scheduler,configured,total,official,aux,x,extra;gc.collect();torch.cuda.empty_cache();return r
