"""Read-only feature geometry and diagnostic summaries; never model alternatives."""
import ast
import inspect
import math
import textwrap
import numpy as np
import torch
from src.evaluation.strict_fpr_diagnostics import percentiles
from src.evaluation.efficientad_branch_audit import parameter_hash,state_hash,EXPECTED_SHA256

SUMMARIES=('OFFICIAL_MEAN','CHANNEL_MAX','CHANNEL_TOP5_MEAN','CHANNEL_TOP10_MEAN')


def pair(value):
    return list(value) if isinstance(value,(list,tuple)) else [value,value]


def geometry(layers,input_shape):
    shape=list(input_shape);jump=[1.,1.];rf=[1.,1.];start=[.5,.5];trace=[]
    for layer in layers:
        if layer['kind'] not in ('Conv2d','AvgPool2d','MaxPool2d'):raise ValueError('Unsupported spatial layer')
        k=pair(layer['kernel']);s=pair(layer['stride']);p=pair(layer['padding']);d=pair(layer.get('dilation',1))
        result=[]
        for axis in range(2):
            effective=(k[axis]-1)*d[axis]+1
            n=math.floor((shape[axis]+2*p[axis]-effective+(s[axis]-1 if layer.get('ceil_mode',False) else 0))/s[axis])+1
            if layer.get('ceil_mode',False) and (n-1)*s[axis]>=shape[axis]+p[axis]:n-=1
            result.append(n)
            start[axis]+=((effective-1)/2-p[axis])*jump[axis]
            rf[axis]+=(effective-1)*jump[axis];jump[axis]*=s[axis]
        shape=result
        if 'observed_output_shape' in layer and shape!=layer['observed_output_shape']:raise ValueError('Calculated/observed layer shape differs')
        trace.append(dict(layer,output_shape=shape.copy(),jump=jump.copy(),rf=rf.copy(),start=start.copy()))
    return dict(input_shape=list(input_shape),output_shape=shape,jump=jump,rf=rf,start=start,layers=trace,
                coordinates='y,x; input pixel i occupies [i,i+1), center i+0.5; RF is bounding support, clipped to actual input for denominator')


def integral(mask):
    return np.pad(np.cumsum(np.cumsum(np.asarray(mask,dtype=np.float64),axis=0),axis=1),((1,0),(1,0)))


def prefix_at(prefix,y,x):
    h,w=np.array(prefix.shape)-1;y=np.clip(y,0,h);x=np.clip(x,0,w)
    iy=np.floor(y).astype(int);ix=np.floor(x).astype(int);dy=y-iy;dx=x-ix
    jy=np.minimum(iy+1,h);jx=np.minimum(ix+1,w)
    return prefix[iy,ix]*(1-dy)*(1-dx)+prefix[jy,ix]*dy*(1-dx)+prefix[iy,jx]*(1-dy)*dx+prefix[jy,jx]*dy*dx


def rectangle_mass(prefix,y0,y1,x0,x1):
    return prefix_at(prefix,y1,x1)-prefix_at(prefix,y0,x1)-prefix_at(prefix,y1,x0)+prefix_at(prefix,y0,x0)


def fractional_mask(mask,size=(256,256)):
    """Exact area overlap of piecewise-constant native pixels; conserves area.

    Unlike adaptive average pooling for non-integer scale ratios, no duplicated
    native boundary pixels. This GT is diagnostic only, never a metric target.
    """
    mask=np.asarray(mask,dtype=np.float64)
    if mask.ndim!=2 or mask.min()<0 or mask.max()>1:raise ValueError('Expected binary/fractional occupancy')
    y=np.linspace(0,mask.shape[0],size[0]+1);x=np.linspace(0,mask.shape[1],size[1]+1)
    area=(mask.shape[0]/size[0])*(mask.shape[1]/size[1])
    out=rectangle_mass(integral(mask),y[:-1,None],y[1:,None],x[None,:-1],x[None,1:])/area
    if out.min()<-1e-10 or out.max()>1+1e-10 or not np.isclose(out.sum()*area,mask.sum(),atol=1e-6):raise ValueError('Fractional mass/range check failed')
    return np.clip(out,0,1)


def context_mask(region,tile_id,core=1056,halo=96):
    if region.shape!=(core,4*core) or tile_id not in range(4):raise ValueError('Core partition mismatch')
    result=np.zeros((core+2*halo,core+2*halo),dtype=np.uint8)
    result[halo:halo+core,halo:halo+core]=region[:,tile_id*core:(tile_id+1)*core]
    return result


def rf_occupancy(fractional,g):
    h,w=fractional.shape;hy,hx=g['output_shape'];cy=g['start'][0]+np.arange(hy)*g['jump'][0];cx=g['start'][1]+np.arange(hx)*g['jump'][1]
    y0=np.clip(cy-g['rf'][0]/2,0,h)[:,None];y1=np.clip(cy+g['rf'][0]/2,0,h)[:,None]
    x0=np.clip(cx-g['rf'][1]/2,0,w)[None,:];x1=np.clip(cx+g['rf'][1]/2,0,w)[None,:]
    area=(y1-y0)*(x1-x0)
    if np.any(area<=0):raise ValueError('RF lacks valid input area')
    occupancy=rectangle_mass(integral(fractional),y0,y1,x0,x1)/area
    if occupancy.min()<-1e-10 or occupancy.max()>1+1e-10:raise ValueError('RF occupancy outside [0,1]')
    hits=(fractional[np.clip(np.floor(cy).astype(int),0,h-1)[:,None],np.clip(np.floor(cx).astype(int),0,w-1)[None,:]]>0)&(cy[:,None]>=0)&(cy[:,None]<h)&(cx[None,:]>=0)&(cx[None,:]<w)
    return np.clip(occupancy,0,1),hits


def channel_partial(discrepancy,occupancy):
    if discrepancy.ndim!=3 or discrepancy.shape[-2:]!=occupancy.shape:raise ValueError('D/occupancy shape mismatch')
    return np.einsum('chw,hw->c',discrepancy,occupancy,dtype=np.float64),float(occupancy.sum())


def merge_channels(parts):
    denominator=sum(p[1] for p in parts)
    if denominator<=0:raise ValueError('No RF overlap for region')
    return sum(p[0] for p in parts)/denominator


def concentration(vector):
    vector=np.asarray(vector,dtype=np.float64)
    if vector.ndim!=1 or not np.isfinite(vector).all() or (vector<0).any():raise ValueError('Invalid nonnegative channel vector')
    c=len(vector);total=float(vector.sum());zero=total<=1e-12;p=vector/total if not zero else np.zeros(c)
    positive=p[p>0];effective=float(np.exp(-np.sum(positive*np.log(positive)))) if not zero else 0.
    ordered=np.sort(p)[::-1]
    if not zero and (abs(p.sum()-1)>1e-12 or not 1-1e-10<=effective<=c+1e-10):raise ValueError('Invalid channel concentration')
    return dict(zero_signal=zero,contributions=p,top1_channel_share=float(ordered[0]),
        **{f'top{int(q*100)}_channel_share':float(ordered[:max(1,math.ceil(c*q))].sum()) for q in (.05,.1,.25)},
        effective_channel_count=effective,effective_channel_fraction=effective/c,channel_max_to_mean_ratio=float(vector.max()/vector.mean()) if not zero else 0.,
        channel_cv=float(vector.std()/vector.mean()) if not zero else 0.)


def map_from_feature(feature,padding,input_shape):
    return torch.nn.functional.interpolate(torch.nn.functional.pad(feature,padding),size=input_shape,mode='bilinear',align_corners=False)


def padding_protocol(model):
    source=inspect.getsource(type(model.model).compute_maps)
    tree=ast.parse(textwrap.dedent(source));pads=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='pad':pads.append(tuple(ast.literal_eval(node.args[1])))
    if not pads or len(set(pads))!=1:raise ValueError('Cannot establish official padding protocol')
    return pads[0] if model.model.pad_maps else (0,0,0,0)


def extract(model,path):
    from anomalib.data.utils import read_image
    if model.training or model.model.training or any(p.requires_grad for p in model.parameters()):raise ValueError('Frozen eval model required')
    captures={};traces={'teacher':[],'student':[]};handles=[]
    def capture(name):
        def hook(module,args,output):captures[name]=output.detach()
        return hook
    def trace(tag,name):
        def hook(module,args,output):
            if isinstance(module,(torch.nn.Conv2d,torch.nn.AvgPool2d,torch.nn.MaxPool2d)):
                stride=module.stride if module.stride is not None else module.kernel_size
                traces[tag].append(dict(name=name,kind=type(module).__name__,kernel=pair(module.kernel_size),stride=pair(stride),padding=pair(module.padding),
                    dilation=pair(getattr(module,'dilation',1)),ceil_mode=bool(getattr(module,'ceil_mode',False)),observed_input_shape=list(args[0].shape[-2:]),observed_output_shape=list(output.shape[-2:])))
            else:raise ValueError('Unknown PDN leaf module; spatial geometry must be inspected')
        return hook
    try:
        for tag in ('teacher','student'):
            pdn=getattr(model.model,tag);handles.append(pdn.register_forward_hook(capture(tag)))
            for name,module in pdn.named_modules():
                if name and not list(module.children()):handles.append(module.register_forward_hook(trace(tag,name)))
        with torch.no_grad():
            image=read_image(path,as_tensor=True)[None].to(next(model.parameters()).device)
            if tuple(image.shape[-2:])!=(1248,1248):raise ValueError('Context shape differs')
            image=model.pre_processor(image)
            if tuple(image.shape[-2:])!=(256,256):raise ValueError('Model input shape differs')
            student,d=model.model.compute_student_teacher_distance(image)
            teacher=captures['teacher'];c=model.model.teacher_out_channels
            if teacher.shape[1]!=c or d.shape[1]!=c or student.shape[1]<c:raise ValueError('Actual ST channel mapping differs')
            teacher_normal=(teacher-model.model.mean_std['mean'])/model.model.mean_std['std'] if model.model.is_set(model.model.mean_std) else teacher
            expected=(teacher_normal-student[:,:c])**2
            formula_error=float((expected-d).abs().max().cpu())
            padding=padding_protocol(model);prepad=d.mean(dim=1,keepdim=True);official,_=model.model.compute_maps(image,student,d,normalize=False)
            raw=map_from_feature(prepad,padding,image.shape[-2:]);error=float((raw-official).abs().max().cpu())
            if not np.isfinite([formula_error,error]).all() or max(formula_error,error)>1e-5:raise ValueError('FAIL_ST_RECONSTRUCTION')
            shapes=dict(teacher_feature_shape=list(teacher.shape),student_feature_shape=list(student.shape),student_ST_feature_shape=list(student[:,:c].shape),
                st_discrepancy_shape=list(d.shape),pre_pad_ST_shape=list(prepad.shape),post_pad_ST_shape=list(torch.nn.functional.pad(prepad,padding).shape),model_map_shape=list(raw.shape))
            g=geometry(traces['teacher'],image.shape[-2:]);gs=geometry(traces['student'],image.shape[-2:])
            if any(g[key]!=gs[key] for key in ('output_shape','jump','rf','start')) or g['output_shape']!=list(d.shape[-2:]):raise ValueError('Teacher/Student RF mismatch')
            diagnostic_features=dict(OFFICIAL_MEAN=prepad,CHANNEL_MAX=d.max(dim=1,keepdim=True).values,
                CHANNEL_TOP5_MEAN=d.topk(max(1,math.ceil(c*.05)),dim=1).values.mean(dim=1,keepdim=True),
                CHANNEL_TOP10_MEAN=d.topk(max(1,math.ceil(c*.10)),dim=1).values.mean(dim=1,keepdim=True))
            maps={name:map_from_feature(value,padding,image.shape[-2:])[0,0].cpu().numpy().copy() for name,value in diagnostic_features.items()}
            result=dict(D=d.cpu().numpy().copy(),maps=maps,shapes=shapes,geometry=g,student_geometry=gs,reconstruction_error=error,discrepancy_formula_error=formula_error,padding=list(padding))
        return result
    finally:
        for handle in handles:handle.remove()


def frozen_control(root):
    from scripts import evaluate_p1_halo96_public as frozen
    from src.data.p1_halo_builder import read_rows,manifest_hash
    if frozen.sha(frozen.CHECKPOINT)!=EXPECTED_SHA256:raise ValueError('FAIL_CHECKPOINT_HASH')
    model=frozen.load_model({},manifest_hash(),torch).to('cuda:0')
    before=parameter_hash(model);state_before=state_hash(model)
    row=next(r for r in read_rows() if r['split']=='test_public')
    result=extract(model,row['context_path'])
    result={key:result[key] for key in ('shapes','geometry','student_geometry','padding','reconstruction_error','discrepancy_formula_error')}
    result.update(parameters_before=before,parameters_after=parameter_hash(model),state_before=state_before,state_after=state_hash(model))
    del model;torch.cuda.empty_cache()
    return result
