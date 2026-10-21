"""Structural PDN audit. Auxiliary backward is diagnostic only; no update step."""
import ast
import gc
import inspect
import math
import random
import textwrap
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from src.evaluation.rectangular_formal_evaluation import (
    ROOT, CHECKPOINT, SHA, TARGET, load_model, checkpoint_gate, input_tensor, input_gate,
    read_json, write_json, write_csv, digest,
)
from src.evaluation.efficientad_branch_audit import parameter_hash, state_hash
from src.evaluation.st_feature_rootcause import geometry, pair

OUT=ROOT/'experiments/phase14a_intermediate_feature_audit'
NATIVE=(1056,4224)
FINAL_CELLS=13888
MEMORY_LIMIT=6500.
RUNTIME_CAVEAT='Single fixed-order measurement, baseline first. Negative overhead does not establish acceleration from hooks; timing variability and order effects were not separated. Runtime is not used to rank candidates.'
PROBE_SCOPE='One auxiliary stage at a time, no optimizer or optimizer state. This is a diagnostic backward peak, not a measurement of joint A+B formal training memory.'

def train_paths():
    folder=ROOT/'datasets/MVTec_AD_2/sheet_metal/train/good'
    paths=sorted(folder.glob('*.png'))
    if len(paths)!=137:raise ValueError('FAIL_NORMAL_TRAIN_DATA')
    return paths

def forward_structure(pdn):
    """Inspect executed self.module order and functional ReLU consumers."""
    source=inspect.getsource(type(pdn).forward)
    tree=ast.parse(textwrap.dedent(source));order=[];relu=[]
    for statement in tree.body[0].body:
        calls=[n for n in ast.walk(statement) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='self']
        if len(calls)>1:raise ValueError('FAIL_UNSUPPORTED_PDN_GRAPH')
        if calls:
            name=calls[0].func.attr;order.append(name)
            if any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='relu' for n in ast.walk(statement)):relu.append(name)
    leaves={n for n,m in pdn.named_modules() if n and not list(m.children())}
    if set(order)!=leaves or len(order)!=len(leaves):raise ValueError('FAIL_PDN_EXECUTION_ORDER')
    links={name:order[order.index(name)+1] for name in relu}
    for name in order:
        if not isinstance(pdn.get_submodule(name),(torch.nn.Conv2d,torch.nn.AvgPool2d,torch.nn.MaxPool2d)):raise ValueError('FAIL_UNSUPPORTED_SPATIAL_MODULE')
    return order,links

def architecture(model):
    result={}
    for tag in ('teacher','student'):
        pdn=getattr(model.model,tag);order,links=forward_structure(pdn)
        result[tag]=dict(module_path='model.'+tag,module_tree=str(pdn),module=type(pdn).__module__,class_name=type(pdn).__name__,
            source_file=inspect.getsourcefile(type(pdn)),source_line=inspect.getsourcelines(type(pdn))[1],signature=str(inspect.signature(type(pdn))),
            class_source=inspect.getsource(type(pdn)),execution_order=order,functional_relu_consumers=links)
    result['official_loss_source']=inspect.getsource(type(model.model).compute_losses)
    result['official_st_source']=inspect.getsource(type(model.model).compute_student_teacher_distance)
    result['normalization_note']='PDN input ImageNet normalization is unchanged. Final teacher mean/std applies to final ST only. Intermediate diagnostic uses raw actual post-ReLU features; no fitted statistics or projector.'
    return result

class Capture:
    """Non-mutating hooks, with explicit original-forward-only retention."""
    def __init__(self,model,wanted=None,metadata=False):
        self.model=model;self.wanted=wanted;self.metadata=metadata
        self.values={};self.records={'teacher':[],'student':[]};self.calls=Counter();self.handles=[]
    def record(self,tag,name,kind,module,output):
        if not isinstance(output,torch.Tensor) or output.ndim!=4:raise ValueError('FAIL_SPATIAL_OUTPUT')
        key=(tag,name);self.calls[key]+=1
        if self.calls[key]!=1:return
        if self.metadata:
            self.records[tag].append(dict(module_name=name,module_type=type(module).__name__ if kind=='module' else 'functional ReLU',
                record_kind=kind,output_shape=list(output.shape[1:]),batch_shape=list(output.shape),dtype=str(output.dtype),requires_grad=output.requires_grad))
        if self.wanted is not None and name in self.wanted.get(tag,[]):
            self.values[key]=output
            if output.requires_grad:output.retain_grad()
    def __enter__(self):
        try:
            for tag in ('teacher','student'):
                pdn=getattr(self.model.model,tag);_,links=forward_structure(pdn)
                for name,module in pdn.named_modules():
                    if self.metadata or (self.wanted is not None and name in self.wanted.get(tag,[])):
                        def hook(m,args,out,t=tag,n=name):self.record(t,n,'module',m,out)
                        self.handles.append(module.register_forward_hook(hook))
                for parent,consumer in links.items():
                    name=parent+'.post_relu'
                    if self.metadata or (self.wanted is not None and name in self.wanted.get(tag,[])):
                        def prehook(m,args,t=tag,n=name):self.record(t,n,'functional_activation',m,args[0])
                        self.handles.append(pdn.get_submodule(consumer).register_forward_pre_hook(prehook))
            return self
        except Exception:
            self.__exit__(None,None,None);raise
    def clear(self):
        self.values.clear();self.calls.clear()
    def __exit__(self,*args):
        for handle in self.handles:handle.remove()
        self.handles.clear()

def inventory(model,x):
    input_gate(x)
    if model.training:raise ValueError('FAIL_INVENTORY_MODE')
    with torch.no_grad():
        official=model.model(x).anomaly_map
        with Capture(model,metadata=True) as capture:hooked=model.model(x).anomaly_map
        error=float((official-hooked).abs().max())
    if error!=0:raise ValueError('FAIL_HOOK_MUTATION')
    result=dict(capture.records,output_max_abs_error=error)
    for tag in ('teacher','student'):
        pdn=getattr(model.model,tag);order,_=forward_structure(pdn);layers=[];g_by_name={}
        records={r['module_name']:r for r in result[tag]}
        for name in order:
            m=pdn.get_submodule(name)
            layers.append(dict(name=name,kind=type(m).__name__,kernel=pair(m.kernel_size),stride=pair(m.stride or m.kernel_size),padding=pair(m.padding),
                dilation=pair(getattr(m,'dilation',1)),ceil_mode=bool(getattr(m,'ceil_mode',False)),observed_output_shape=records[name]['output_shape'][-2:]))
            g_by_name[name]=geometry(layers,TARGET)
        result[tag+'_geometry']=g_by_name
    return result

def natural_match(teacher,student):return list(teacher)==list(student)
def feature_bytes(teacher,student):return int((math.prod(teacher)+math.prod(student))*4)

def make_stages(inv):
    # Consecutive identical spatial sizes form one stage. Prefer its last
    # Conv/activation; a pool-only resolution is retained as a pool-only stage.
    # This preserves every meaningful spatial output without inventing layers.
    branches={}
    for tag in ('teacher','student'):
        groups=[]
        for r in inv[tag]:
            if not r['module_name']:continue
            hw=r['output_shape'][-2:]
            if not groups or groups[-1][0]['output_shape'][-2:]!=hw:groups.append([])
            groups[-1].append(r)
        reps=[]
        for group in groups:
            preferred=[r for r in group if r['module_type'] in ('Conv2d','functional ReLU')]
            reps.append((preferred or group)[-1])
        branches[tag]=reps
    if len(branches['teacher'])!=len(branches['student']):raise ValueError('FAIL_STAGE_ALIGNMENT')
    stages=[];scale=[n/t for n,t in zip(NATIVE,TARGET)]
    for depth,(t,s) in enumerate(zip(branches['teacher'],branches['student'])):
        g=inv['teacher_geometry'][t['module_name'].split('.')[0]];gs=inv['student_geometry'][s['module_name'].split('.')[0]]
        if any(g[k]!=gs[k] for k in ('output_shape','jump','rf','start')):raise ValueError('FAIL_RF_ALIGNMENT')
        c,h,w=t['output_shape'];final=depth==len(branches['teacher'])-1;match=natural_match(t['output_shape'],s['output_shape'])
        if final and (t['output_shape']!=[384,56,248] or s['output_shape']!=[768,56,248]):raise ValueError('FAIL_FINAL_REFERENCE')
        stages.append(dict(stage_id='S'+str(depth+1),depth=depth,teacher_module=t['module_name'],student_module=s['module_name'],
            stage_kind=t['module_type'],representative_note='Last Conv/activation at this resolution' if t['module_type'] in ('Conv2d','functional ReLU') else 'Pool-only resolution; no Conv/activation exists at this exact size',
            shape=t['output_shape'],teacher_shape=t['output_shape'],student_shape=s['output_shape'],natural_match=match,C=c,H=h,W=w,cells=h*w,cell_ratio_vs_final=h*w/FINAL_CELLS,
            stride_model=g['jump'],RF_model=g['rf'],first_cell_center=g['start'],stride_native=[a*b for a,b in zip(g['jump'],scale)],RF_native=[a*b for a,b in zip(g['rf'],scale)],native_stride=[a*b for a,b in zip(g['jump'],scale)],native_RF=[a*b for a,b in zip(g['rf'],scale)],
            feature_bytes=feature_bytes(t['output_shape'],s['output_shape']),teacher_feature_bytes=math.prod(t['output_shape'])*4,student_feature_bytes=math.prod(s['output_shape'])*4,
            minimum_retained_activation_mib=feature_bytes(t['output_shape'],s['output_shape'])/1024**2,
            memory_bound_note='Only two retained feature tensors, batch1 FP32. Excludes autograd intermediates, discrepancy, gradients, and allocator workspace.',
            candidate_valid=bool(match and not final and h*w>FINAL_CELLS),projector_required=False,final_reference=final,
            official_final_student_st_shape=[384,56,248] if final else None,official_final_channel_rule='Official first384 only; never applied to intermediate stages' if final else 'All channels naturally match; no slicing'))
    return stages

def select_candidates(stages,final_cells=FINAL_CELLS):
    valid=[s for s in stages if s['candidate_valid'] and s['training_feasible'] and s['gradient_finite'] and s['gradient_probe_peak_vram']<=MEMORY_LIMIT]
    if not valid:return None,None
    a=min(valid,key=lambda s:(-s['cells'],s['C'],-s['depth']))
    others=[s for s in valid if s['cells']!=a['cells']]
    midpoint=(math.log(a['cells'])+math.log(final_cells))/2
    b=min(others,key=lambda s:(abs(math.log(s['cells'])-midpoint),s['C'],-s['depth'])) if others else None
    return a,b

def imagenette_batch(model):
    if not (ROOT/'datasets/imagenette').is_dir():raise ValueError('FAIL_IMAGENETTE_NOT_PRESENT')
    model.prepare_imagenette_data(TARGET)
    extra=next(iter(model.imagenet_loader))[0].to('cuda:0');input_gate(extra)
    return extra

def seed():
    random.seed(42);np.random.seed(42);torch.manual_seed(42)

def gradient_probe(model,x,extra,stage):
    if not stage['candidate_valid']:raise ValueError('FAIL_INVALID_CANDIDATE')
    seed();model.zero_grad(set_to_none=True)
    before=parameter_hash(model);state_before=state_hash(model)
    training_modes={n:m.training for n,m in model.named_modules()}
    result=dict(stage_id=stage['stage_id'],parameters_before=before,state_before=state_before,optimizer_steps=0,aux_coefficient=1.0,
        loss_definition='Official ST+AE+STAE including ImageNette penalty, plus mean((T_stage.detach()-S_stage)^2). Diagnostic backward only.',probe_scope=PROBE_SCOPE,oom=False)
    capture=None;losses=None;aux=None;total=None;t=None;s=None
    try:
        model.train();gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        with Capture(model,{'teacher':[stage['teacher_module']],'student':[stage['student_module']]}) as capture:
            losses=model.model(batch=x,batch_imagenet=extra)
            if not isinstance(losses,tuple) or len(losses)!=3:raise ValueError('FAIL_OFFICIAL_LOSS_API')
            t=capture.values['teacher',stage['teacher_module']];s=capture.values['student',stage['student_module']]
            if t.shape!=s.shape:raise ValueError('FAIL_NATURAL_MATCH')
            aux=(t.detach()-s).square().mean();total=sum(losses)+aux
            numbers=[float(v.detach()) for v in (*losses,aux,total)]
            if not np.isfinite(numbers).all():raise ValueError('FAIL_NONFINITE_LOSS')
            total.backward();torch.cuda.synchronize()
            if s.grad is None:raise ValueError('FAIL_MISSING_STAGE_GRADIENT')
            finite=bool(torch.isfinite(s.grad).all()) and all(bool(torch.isfinite(p.grad).all()) for p in model.model.student.parameters() if p.grad is not None)
            norm=float(torch.linalg.vector_norm(s.grad))
            if not finite or not math.isfinite(norm):raise ValueError('FAIL_NONFINITE_GRADIENT')
            result.update(losses=dict(zip(('ST','AE','STAE','aux','total'),numbers)),gradient_finite=finite,gradient_norm=norm,
                teacher_hook_calls=capture.calls['teacher',stage['teacher_module']],student_hook_calls=capture.calls['student',stage['student_module']],
                captured_original_only=True,teacher_parameter_gradients_absent=all(p.grad is None for p in model.model.teacher.parameters()),
                peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
                NaN=False,Inf=False)
            if norm<=0:raise ValueError('FAIL_ZERO_STAGE_GRADIENT')
    except torch.cuda.OutOfMemoryError:
        result.update(oom=True,gradient_finite=False,gradient_norm=None,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
    finally:
        model.zero_grad(set_to_none=True)
        if capture is not None:capture.clear()
        losses=aux=total=t=s=None
        for n,m in model.named_modules():m.training=training_modes[n]
        gc.collect();torch.cuda.empty_cache()
        result.update(parameters_after=parameter_hash(model),state_after=state_hash(model),all_gradients_cleared=all(p.grad is None for p in model.parameters()))
        if result['parameters_after']!=before or result['state_after']!=state_before:raise ValueError('FAIL_PARAMETER_STATE_MUTATION')
    result['training_feasible']=bool(not result['oom'] and result['gradient_finite'] and result['peak_allocated_mib']<=MEMORY_LIMIT)
    return result

def timed(fn):
    torch.cuda.synchronize();start=time.perf_counter();out=fn();torch.cuda.synchronize()
    return out,(time.perf_counter()-start)*1000

def benchmark_loop(fn,warmup=20,count=100):
    for _ in range(warmup):fn()
    values=[]
    for _ in range(count):out,ms=timed(fn);values.append(ms);del out
    return dict(mean_ms=float(np.mean(values)),median_ms=float(np.median(values)),std_ms=float(np.std(values)),count=count,warmup=warmup,samples_ms=values)

def runtime_probe(model,x,stages):
    model.eval();seed();results={}
    with torch.no_grad():
        results['baseline']=benchmark_loop(lambda:model.model(x))
        for stage in stages:
            if not stage['candidate_valid']:continue
            with Capture(model,{'teacher':[stage['teacher_module']],'student':[stage['student_module']]}) as c:
                def forward():
                    c.clear();official=model.model(x)
                    t=c.values['teacher',stage['teacher_module']];s=c.values['student',stage['student_module']]
                    discrepancy=(t-s).square().mean(dim=1,keepdim=True)
                    return official,discrepancy
                results[stage['stage_id']]=benchmark_loop(forward)
                c.clear()
            v=results[stage['stage_id']];base=results['baseline']['mean_ms']
            v.update(overhead_ms=v['mean_ms']-base,overhead_percent=(v['mean_ms']/base-1)*100)
            print(f"Runtime {stage['stage_id']}: {v['mean_ms']:.4f} ms; overhead {v['overhead_percent']:.2f}%",flush=True)
    return dict(paths=results,normal_train_sample_only=True,warmup=20,formal_forwards_per_path=100,measurement_caveat=RUNTIME_CAVEAT,
        interpretation='Diagnostic overhead includes capture bookkeeping and intermediate discrepancy/reduction only. Hooks installed outside timing. No new backbone forward, no new convolution; not a detection method benchmark.',
        fixed_order=['baseline']+[s['stage_id'] for s in stages if s['candidate_valid']])

def protocol_gate():
    for path in (Path(__file__),ROOT/'scripts/audit_rect_intermediate_features.py'):
        tree=ast.parse(path.read_text(encoding='utf-8'))
        for n in ast.walk(tree):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in {'step','fit','training_step','configure_optimizers','test','validate','native','integrate','pro_at_fprs','get_mask','get_score','read_membership','references'}:raise ValueError('FAIL_FORBIDDEN_EXECUTION_PATH')
    return True
