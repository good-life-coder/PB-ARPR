"""Performance-blind architecture fallback over the immutable PB-ARPR rule."""
import ast
import gc
import hashlib
import inspect
import json
import math
from src.common.pbarpr_geometry import select_rect as old_select

def frozen_config():
    tree=ast.parse(inspect.getsource(old_select));maximum=[];target=[];multiple=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Compare) and isinstance(node.ops[0],ast.LtE) and isinstance(node.comparators[0],ast.Constant):maximum.append(node.comparators[0].value)
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Div) and isinstance(node.right,ast.Constant):target.append(node.right.value)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='range':multiple.append(ast.literal_eval(node.args[2]))
    if len(set(maximum))!=1 or len(set(target))!=1 or len(set(multiple))!=1:raise ValueError('FAIL_FROZEN_GEOMETRY_CONFIG')
    return dict(maximum_pixels=maximum[0],target_pixels=target[0],multiple=multiple[0])

def candidates():
    c=frozen_config();step=c['multiple'];budget=c['maximum_pixels']
    return [(h,w) for h in range(step,budget//step+1,step) for w in range(step,budget//h+1,step)]

def canonical_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def choose(candidates_hw,native_ratio):
    if not candidates_hw:raise ValueError('FAIL_NO_FEASIBLE_GEOMETRY')
    error={hw:abs(math.log((hw[0]/hw[1])/native_ratio)) for hw in candidates_hw};minimum=min(error.values());target=frozen_config()['target_pixels']
    ties=[hw for hw in candidates_hw if error[hw]-minimum<=1e-9]
    return min(ties,key=lambda hw:(abs(math.prod(hw)-target),-math.prod(hw),hw[0],hw[1]))

def select(native_ratio,checker,candidate_hw=None):
    raw=old_select(native_ratio);records={raw:checker(*raw)}
    if records[raw]['feasible']:return raw,records
    for hw in candidates() if candidate_hw is None else candidate_hw:
        if hw not in records:records[hw]=checker(*hw)
    return choose([hw for hw,r in records.items() if r['feasible']],native_ratio),records

def feasibility_record(H,W):
    import torch
    from src.evaluation import ksdd2_training_calibration as cal
    r=dict(H=H,W=W,pixels=H*W,feasible=False,full_forward_attempted=False,data_source='synthetic_zero',failure_stage=None,failure_reason=None,
        teacher_shape=None,student_shape=None,ae_latent_shape=None,ae_encoder_input_shape=None,ae_output_shape=None,anomaly_map_shape=None)
    m=None;hooks=[];stage=['initialization']
    try:
        m=cal.fresh((H,W));r['parameters']=sum(v.numel() for v in m.parameters())
        def before(name):
            def fn(module,args):stage[0]=name
            return fn
        def capture(key):
            def fn(module,args,value):r[key]=list(value.shape)
            return fn
        for name,mod,key in [('Teacher',m.model.teacher,'teacher_shape'),('Student',m.model.student,'student_shape'),('AE',m.model.ae,'ae_output_shape')]:
            hooks.extend([mod.register_forward_pre_hook(before(name)),mod.register_forward_hook(capture(key))])
        hooks.append(m.model.ae.encoder.register_forward_hook(capture('ae_latent_shape')))
        def bottleneck(module,args):r['ae_encoder_input_shape']=list(args[0].shape)
        hooks.append(m.model.ae.encoder.enconv6.register_forward_pre_hook(bottleneck))
        with torch.no_grad():
            x=m.pre_processor(torch.zeros(1,3,H,W,device='cuda',dtype=torch.float32));r['full_forward_attempted']=True
            result=m.model(x);anomaly=result.anomaly_map
            r['anomaly_map_shape']=list(anomaly.shape)
            if anomaly.shape!=(1,1,H,W) or not torch.isfinite(anomaly).all():raise ValueError('FAIL_ANOMALY_MAP')
            if r['teacher_shape'][1:]!=[384,*r['student_shape'][2:]] or r['ae_output_shape']!=r['teacher_shape']:raise ValueError('FAIL_NATURAL_MATCH')
            r['feasible']=True
            del x,result,anomaly
    except torch.cuda.OutOfMemoryError:raise
    except (RuntimeError,ValueError) as exc:r.update(failure_stage=stage[0],failure_reason=str(exc))
    finally:
        for h in hooks:h.remove()
        del m;gc.collect();torch.cuda.empty_cache()
    return r

def is_efficientad_geometry_feasible(H,W):return feasibility_record(H,W)['feasible']
