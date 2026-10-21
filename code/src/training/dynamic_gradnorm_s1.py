"""Detached first-order gradient-norm control; original S1 and official losses."""
import math
import time
import torch
from src.models import multiscale_st_s1 as s1

RHOS=(.5,1.)
BETA=.90
LOW=.005
HIGH=5.
SHARED=['conv1.bias','conv1.weight']


class Controller:
    def __init__(self,rho):
        if rho not in RHOS: raise ValueError('FAIL_UNREGISTERED_RHO')
        self.rho=rho;self.log_ema=None;self.steps=0;self.last={}
        self.lower_boundary_hits=0;self.upper_boundary_hits=0

    def state(self):return dict(rho=self.rho,log_ema=self.log_ema,steps=self.steps,last=dict(self.last))

    def state_dict(self):
        return dict(version=1,rho=self.rho,beta=BETA,lambda_min=LOW,lambda_max=HIGH,
                    log_lambda_ema=self.log_ema,lambda_ema=self.last.get('lambda_ema'),
                    lambda_used=self.last.get('lambda_used'),controller_step=self.steps,
                    total_controller_updates=self.steps,lower_boundary_hits=self.lower_boundary_hits,
                    upper_boundary_hits=self.upper_boundary_hits,
                    boundary_hit_count=self.lower_boundary_hits+self.upper_boundary_hits,last=dict(self.last))

    def load_state_dict(self,state):
        expected=dict(version=1,rho=self.rho,beta=BETA,lambda_min=LOW,lambda_max=HIGH)
        if any(state.get(k)!=v for k,v in expected.items()):raise ValueError('FAIL_CONTROLLER_PROTOCOL')
        count=state.get('controller_step')
        if type(count) is not int or count<0 or count!=state.get('total_controller_updates'):
            raise ValueError('FAIL_CONTROLLER_STEP')
        lo,hi=state.get('lower_boundary_hits'),state.get('upper_boundary_hits')
        if any(type(v) is not int or v<0 for v in (lo,hi)) or lo+hi>count or lo+hi!=state.get('boundary_hit_count'):
            raise ValueError('FAIL_CONTROLLER_COUNTERS')
        last=state.get('last',{});log=state.get('log_lambda_ema')
        if count:
            if not isinstance(log,(int,float)) or not math.isfinite(log):raise ValueError('FAIL_CONTROLLER_EMA')
            if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in last.values()):raise ValueError('FAIL_CONTROLLER_FINITE')
            ema=math.exp(log);used=min(HIGH,max(LOW,ema))
            if ema<=0 or any(state.get(k)!=v or last.get(k)!=v for k,v in [('lambda_ema',ema),('lambda_used',used)]):
                raise ValueError('FAIL_CONTROLLER_EMA')
        elif log is not None or last or state.get('lambda_ema') is not None or state.get('lambda_used') is not None:
            raise ValueError('FAIL_CONTROLLER_INITIAL_STATE')
        self.log_ema=log;self.steps=count;self.last=dict(last)
        self.lower_boundary_hits=lo;self.upper_boundary_hits=hi

    def update(self,n_off,n_s1,*,split,normal):
        if split!='train' or not normal:raise ValueError('FAIL_CONTROLLER_SOURCE')
        off,aux=float(n_off.detach()),float(n_s1.detach())
        if not all(math.isfinite(v) and v>0 for v in (off,aux)):raise ValueError('FAIL_CONTROLLER_NORMS')
        raw=self.rho*off/(aux+1e-12)
        self.log_ema=math.log(raw) if self.log_ema is None else BETA*self.log_ema+(1-BETA)*math.log(raw)
        ema=math.exp(self.log_ema);used=min(HIGH,max(LOW,ema))
        self.lower_boundary_hits+=int(ema<=LOW);self.upper_boundary_hits+=int(ema>=HIGH)
        self.steps+=1
        self.last=dict(lambda_raw=raw,lambda_ema=ema,lambda_used=used,
                       boundary_hit=ema<=LOW or ema>=HIGH,n_official=off,n_s1=aux,
                       expected_ratio=used*aux/off)
        return dict(self.last)


def shared_parameters(student):
    named=dict(student.named_parameters())
    params=[named[n] for n in SHARED]
    if sum(p.numel() for p in params)!=6272:raise ValueError('FAIL_SHARED_NUMEL')
    return list(SHARED),params


def gradient_norms(official,shallow,params):
    go=torch.autograd.grad(official,params,retain_graph=True,create_graph=False)
    gs=torch.autograd.grad(shallow,params,retain_graph=True,create_graph=False)
    off=torch.cat([v.detach().reshape(-1).double() for v in go])
    aux=torch.cat([v.detach().reshape(-1).double() for v in gs])
    if not bool(torch.isfinite(off).all() and torch.isfinite(aux).all()):raise ValueError('FAIL_NONFINITE_GRADIENT')
    return off.norm(),aux.norm(),off,aux


def confirm_shared(student,official,shallow):
    pairs=sorted(student.named_parameters());params=[p for _,p in pairs]
    support=[]
    for loss in (official,shallow):
        gs=torch.autograd.grad(loss,params,retain_graph=True,create_graph=False,allow_unused=True)
        support.append({n for (n,p),g in zip(pairs,gs) if g is not None and bool(torch.count_nonzero(g))})
    shared=sorted(support[0]&support[1])
    if shared!=SHARED or sorted(support[1])!=SHARED:raise ValueError('FAIL_AUTOGRAD_SHARED_SET')
    names,params=shared_parameters(student)
    return dict(names=names,numel=sum(p.numel() for p in params),autograd_confirmed=True)


def total_loss(official,shallow,coefficient):
    return official+torch.as_tensor(coefficient,device=shallow.device,dtype=shallow.dtype).detach()*shallow


def loss_components(model,x,extra):
    """Same forward, hooks and helper calls as MultiScaleSTS1.p2_loss; no loss redefinition."""
    s1.input_gate(x);s1.input_gate(extra)
    if not model.model.training:raise ValueError('FAIL_LOSS_MODE')
    penalty={};calls=0
    def observe(module,args,output):
        nonlocal calls
        calls+=1
        if calls==2:
            with torch.no_grad():penalty['value']=float(output[:,:model.model.teacher_out_channels].square().mean())
    handle=model.model.student.register_forward_hook(observe)
    try:
        with s1.Capture(model,s1.WANTED) as capture:
            st,ae,stae=model.model(batch=x, batch_imagenet=extra)
        t,s=capture.values['teacher',s1.S1],capture.values['student',s1.S1]
        if tuple(t.shape)!=(1,128,253,1021) or s.shape!=t.shape:raise ValueError('FAIL_S1_SHAPE')
        shallow=s1.hard_loss(s1.discrepancy(t, s))
    finally:handle.remove()
    if calls!=3 or 'value' not in penalty:raise ValueError('FAIL_OFFICIAL_API')
    official=st+ae+stae
    if not all(bool(torch.isfinite(v)) for v in (official,shallow,st,ae,stae)):raise ValueError('FAIL_NONFINITE_LOSS')
    logs=dict(L_official=float(official.detach()),final_st=float(st.detach())-penalty['value'],
              ae=float(ae.detach()),stae=float(stae.detach()),L_s1_raw=float(shallow.detach()),
              imagenette_penalty=penalty['value'],student_forward_calls=calls)
    return official,shallow,logs


class DynamicS1(s1.MultiScaleSTS1):
    """Identical inference modules; only training_step/loss weighting changes."""
    def __init__(self,rho,**kwargs):
        super().__init__(**kwargs)
        self.controller=Controller(rho)
        self.shared_audit=None

    def training_step(self,batch,batch_idx,*args,**kwargs):
        if not bool((batch.gt_label==0).all()):raise ValueError('FAIL_NONNORMAL_TRAIN_BATCH')
        return super().training_step(batch,batch_idx,*args,**kwargs)

    def p2_loss(self,x,imagenette):
        official,shallow,logs=loss_components(self,x,imagenette)
        if self.shared_audit is None:self.shared_audit=confirm_shared(self.model.student,official,shallow)
        _,params=shared_parameters(self.model.student)
        torch.cuda.synchronize();started=time.perf_counter()
        off,aux,_,_=gradient_norms(official,shallow,params)
        values=self.controller.update(off,aux,split='train',normal=True)
        torch.cuda.synchronize();seconds=time.perf_counter()-started
        total=total_loss(official,shallow,values['lambda_used'])
        if not bool(torch.isfinite(total)):raise ValueError('FAIL_NONFINITE_TOTAL')
        with torch.no_grad():self.lambda_s1.fill_(values['lambda_used'])
        weighted=float((values['lambda_used']*shallow).detach());number=float(total.detach())
        logs.update(rho=self.controller.rho,**values,weighted_s1_loss=weighted,total_loss=number,
                    aux_fraction=weighted/number,controller_seconds=seconds)
        return total,logs
