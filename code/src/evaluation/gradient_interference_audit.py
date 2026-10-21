"""Frozen S1 gradient geometry; no update or evaluation-metric execution."""
import ast
import gc
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from src.models import multiscale_st_s1 as s1
from src.evaluation import multiscale_st_calibration as calibration
from src.evaluation import rectangular_formal_evaluation as rect
from src.evaluation import multiscale_st_s1_evaluation as p2
from src.evaluation.intermediate_feature_audit import Capture
from src.evaluation.efficientad_branch_audit import parameter_hash, state_hash

LAMBDA=1.8656724688129778
ROOT=calibration.ROOT
OUT=ROOT/'experiments/phase14c_a_gradient_interference'

def save(name,value):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def hash_gate(path,expected):
    actual=rect.digest(path)
    if actual!=expected: raise ValueError('FAIL_CHECKPOINT_SHA')
    return actual

def select_samples(paths,split,labels):
    expected=137 if split=='train' else 19 if split=='validation' else -1
    if len(paths)!=expected or len(labels)!=expected or any(labels): raise ValueError('FAIL_NORMAL_SOURCE')
    return paths[:64] if split=='train' else paths

def protocol_gate():
    for path in (Path(__file__),ROOT/'scripts/audit_s1_gradient_interference.py'):
        for n in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(n,ast.Call):
                name=n.func.attr if isinstance(n.func,ast.Attribute) else getattr(n.func,'id','')
                if name.lower() in {'step','fit','training_step','configure_optimizers','test','predict','metrics','auroc','aupro'}: raise ValueError('FAIL_FORBIDDEN_PATH')
    return True

def flatten(values,names=None):
    return torch.cat([values[n].reshape(-1).double() for n in sorted(names or values)])

def nonzero_names(values): return sorted(n for n,v in values.items() if bool(torch.count_nonzero(v)))

def extract_gradients(module,losses):
    params=sorted((n,p) for n,p in module.named_parameters() if p.requires_grad)
    result={}
    for i,(key,loss) in enumerate(losses.items()):
        module.zero_grad(set_to_none=True)
        grads=torch.autograd.grad(loss,[p for _,p in params],retain_graph=i<len(losses)-1,allow_unused=True)
        result[key]={n:(g.detach().cpu().clone() if g is not None else torch.zeros_like(p,device='cpu')) for (n,p),g in zip(params,grads)}
        if any(not bool(torch.isfinite(v).all()) for v in result[key].values()): raise ValueError('FAIL_NONFINITE_GRADIENT')
    module.zero_grad(set_to_none=True)
    return result

def cosine(a,b):
    return float(torch.dot(a,b)/(torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b)))

def gradient_metrics(official,final,shallow):
    o,f,s=[x.detach().cpu().double().flatten() for x in (official,final,shallow)]
    no,nf,ns=[float(torch.linalg.vector_norm(x)) for x in (o,f,s)]
    if min(no,nf,ns)<=0: raise ValueError('FAIL_ZERO_GRADIENT')
    total=o+LAMBDA*s; dot=float(torch.dot(o,s))
    projected=s-dot/(no**2+1e-12)*o if dot<0 else s
    c=cosine(total,o); pc=cosine(o+LAMBDA*projected,o)
    return dict(cos_official_s1=cosine(o,s),cos_final_s1=cosine(f,s),norm_official=no,norm_final=nf,norm_s1=ns,
        weighted_norm_s1=LAMBDA*ns,weighted_ratio_vs_official=LAMBDA*ns/(no+1e-12),weighted_ratio_vs_final=LAMBDA*ns/(nf+1e-12),
        cos_p2_vs_official=c,angle_p2_vs_official_deg=float(np.degrees(np.arccos(np.clip(c,-1,1)))),norm_p2_ratio=float(torch.linalg.vector_norm(total))/no,
        lambda_equal_norm=no/(ns+1e-12),projected_cosine=pc,projected_angle_deg=float(np.degrees(np.arccos(np.clip(pc,-1,1)))),projected_dot_official=float(torch.dot(projected,o)))

def summary(values):
    a=np.asarray(values,float)
    return dict(mean=float(a.mean()),median=float(np.median(a)),p25=float(np.percentile(a,25)),p75=float(np.percentile(a,75)))

def classify(rows):
    cos=np.array([r['cos_official_s1'] for r in rows]); ratio=np.array([r['weighted_ratio_vs_official'] for r in rows])
    conflict=bool(np.median(cos)<0 or np.mean(cos<-.25)>=.25)
    dominance=bool(np.median(ratio)>=2 or np.mean(ratio>=2)>=.5)
    mechanism='BOTH_CONFLICT_AND_DOMINANCE' if conflict and dominance else 'DIRECTION_CONFLICT_ONLY' if conflict else 'MAGNITUDE_DOMINANCE_ONLY' if dominance else 'NEITHER'
    return dict(direction_conflict=conflict,magnitude_dominance=dominance,mechanism=mechanism,next_experiment='GRADIENT_CONFLICT_HANDLING' if conflict else 'GRADNORM_BALANCED_S1' if dominance else 'S3_AUXILIARY_TRIAL')

def drift(a,b):
    a=a.detach().cpu().double().flatten(); b=b.detach().cpu().double().flatten()
    return dict(relative_l2=float(torch.linalg.vector_norm(b-a)/(torch.linalg.vector_norm(a)+1e-12)),cosine=cosine(a,b),absolute_norm_difference=abs(float(torch.linalg.vector_norm(b)-torch.linalg.vector_norm(a))))

def fresh_audit_model():
    model,meta=calibration.fresh_model()
    initial=parameter_hash(model)
    branch_hashes={k:parameter_hash(getattr(model.model,k)) for k in ('teacher','student','ae')}
    # Recreate separately to verify seed42 Student/AE initialization reproducibility.
    other,other_meta=calibration.fresh_model()
    if initial!=parameter_hash(other): raise ValueError('FAIL_INITIALIZATION_REPRODUCIBILITY')
    del other; gc.collect(); torch.cuda.empty_cache()
    model.prepare_imagenette_data(s1.TARGET)
    paths=calibration.normal_paths('train')
    class NormalBatches:
        def __len__(self): return len(paths)
        def __iter__(self):
            for path in paths: yield SimpleNamespace(image=calibration.normal_input(model,path))
    with torch.no_grad(): model.model.mean_std.update(model.teacher_channel_mean_std(NormalBatches()))
    if branch_hashes!={k:parameter_hash(getattr(model.model,k)) for k in branch_hashes}: raise ValueError('FAIL_INITIALIZATION_PARAMETER_MUTATION')
    model.train()
    meta.update(seed_reproduction=True,teacher_statistics_samples=137,teacher_statistics_before_audit=True)
    return model,meta

def sample_gradients(model,path,extra):
    model.zero_grad(set_to_none=True)
    x=calibration.normal_input(model,path)
    t,s,student,distance=s1.s1_features(model,x)
    # Same official compute_losses as forward; no second augmentation/penalty.
    official_losses=model.model.compute_losses(x,extra,distance)
    final=s1.hard_loss(distance)
    shallow=s1.hard_loss(s1.discrepancy(t,s))
    losses=dict(official=sum(official_losses),final=final,s1=shallow)
    numbers={k:float(v.detach()) for k,v in losses.items()}
    if not np.isfinite(list(numbers.values())).all(): raise ValueError('FAIL_LOSS')
    grads=extract_gradients(model.model.student,losses)
    if any(p.grad is not None for p in model.model.teacher.parameters()): raise ValueError('FAIL_TEACHER_GRAD')
    affected=nonzero_names(grads['s1']); shared=sorted(set(affected)&set(nonzero_names(grads['official'])))
    if not shared: raise ValueError('FAIL_SHARED_PARAMS')
    row=gradient_metrics(*(flatten(grads[k],shared) for k in ('official','final','s1')))
    row.update(sample_id=path.stem,path=str(path),**{k+'_loss':v for k,v in numbers.items()})
    layers=[]
    for layer in sorted(set(n.split('.')[0] for n in shared)):
        names=[n for n in shared if n.split('.')[0]==layer]
        layers.append(dict(sample_id=path.stem,layer=layer,**gradient_metrics(*(flatten(grads[k],names) for k in ('official','final','s1')))))
    return row,layers,affected,shared,{n:grads['s1'][n].numel() for n in shared}

def gradient_run(precheck=False):
    model,meta=fresh_audit_model()
    before=parameter_hash(model); state_before=state_hash(model)
    paths=select_samples(calibration.normal_paths('train'),'train',[0]*137)
    iterator=iter(model.imagenet_loader); rows=[]; layers=[]; sets=[]
    for i,path in enumerate(paths[:1] if precheck else paths):
        try: extra=next(iterator)[0].to(model.device)
        except StopIteration:
            iterator=iter(model.imagenet_loader); extra=next(iterator)[0].to(model.device)
        row,lr,affected,shared,numel=sample_gradients(model,path,extra)
        rows.append(row);layers.extend(lr);sets.append(dict(sample_id=path.stem,affected=affected,shared=shared,numel=numel))
        print(f'GRADIENT {i+1}/{1 if precheck else 64}: cosine={row["cos_official_s1"]:.6f}, weighted ratio={row["weighted_ratio_vs_official"]:.6f}',flush=True)
    model.zero_grad(set_to_none=True)
    hashes=dict(parameters_before=before,parameters_after=parameter_hash(model),state_before=state_before,state_after=state_hash(model))
    if hashes['parameters_after']!=before or hashes['state_after']!=state_before: raise ValueError('FAIL_MUTATION')
    if any(s['shared']!=sets[0]['shared'] for s in sets): raise ValueError('FAIL_VARYING_SHARED_SET')
    result=dict(initialization=meta,hashes=hashes,sample_ids=[p.stem for p in paths],parameter_sets=sets,shared=sets[0]['shared'],shared_numel=sum(sets[0]['numel'].values()),no_optimizer=True,teacher_grad_absent=True)
    del model;gc.collect();torch.cuda.empty_cache()
    return rows,layers,result

@torch.no_grad()
def checkpoint_drift():
    baseline,bmeta=rect.load_model(); trained,pmeta=p2.load_model()
    models={'Rect':baseline,'P2':trained}
    before={k:(parameter_hash(m),state_hash(m)) for k,m in models.items()}
    parameters=[]
    for layer in ('conv1','conv2','conv3','conv4'):
        a=dict(baseline.model.student.get_submodule(layer).named_parameters());b=dict(trained.model.student.get_submodule(layer).named_parameters())
        parameters.append(dict(layer=layer,**drift(flatten(a),flatten(b))))
    paths=select_samples(calibration.normal_paths('validation'),'validation',[0]*19)
    features=[]
    wanted={k:['conv1.post_relu','conv2.post_relu','conv4'] for k in ('teacher','student')}
    for i,path in enumerate(paths):
        values={}
        for tag,model in models.items():
            x=calibration.normal_input(model,path)
            with Capture(model,wanted) as capture: model.model.compute_student_teacher_distance(x)
            values[tag]={}
            for stage,name in [('S1','conv1.post_relu'),('S3','conv2.post_relu'),('Final','conv4')]:
                t=capture.values['teacher',name];s=capture.values['student',name]
                if stage=='Final':
                    s=s[:,:model.model.teacher_out_channels]
                    t=(t-model.model.mean_std['mean'])/model.model.mean_std['std']
                values[tag][stage]=(s.cpu(),float((t-s).square().mean()))
        for stage in ('S1','S3','Final'):
            a,da=values['Rect'][stage]; b,db=values['P2'][stage]
            features.append(dict(sample_id=path.stem,stage=stage,rect_discrepancy=da,p2_discrepancy=db,**drift(a,b)))
        print(f'NORMAL FEATURE {i+1}/19',flush=True)
    after={k:(parameter_hash(m),state_hash(m)) for k,m in models.items()}
    if before!=after: raise ValueError('FAIL_CHECKPOINT_MODEL_MUTATION')
    means={stage:{key:float(np.mean([r[key] for r in features if r['stage']==stage])) for key in ('rect_discrepancy','p2_discrepancy','cosine','relative_l2')} for stage in ('S1','S3','Final')}
    tradeoff=means['S1']['p2_discrepancy']<means['S1']['rect_discrepancy'] and means['Final']['p2_discrepancy']>means['Final']['rect_discrepancy']
    return parameters,features,dict(means=means,shallow_deep_tradeoff=bool(tradeoff),hashes_before=before,hashes_after=after,strict_loads=[bmeta,pmeta],source='19 normal validation; intermediate raw, Final official teacher-normalized')

def figures(rows,parameters,features):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    specs=[('gradient_cosine_distribution','cos_official_s1'),('weighted_gradient_ratio','weighted_ratio_vs_official'),('resultant_angle_distribution','angle_p2_vs_official_deg'),('lambda_gradnorm_distribution','lambda_equal_norm')]
    for name,key in specs:
        fig,ax=plt.subplots();ax.hist([r[key] for r in rows],bins=16);ax.set_xlabel(key);ax.set_ylabel('Normal train samples');fig.tight_layout();fig.savefig(folder/(name+'.png'));plt.close(fig)
    fig,ax=plt.subplots();ax.bar([r['layer'] for r in parameters],[r['relative_l2'] for r in parameters]);ax.set_ylabel('Relative L2 drift');fig.tight_layout();fig.savefig(folder/'layer_parameter_drift.png');plt.close(fig)
    fig,ax=plt.subplots(); x=np.arange(3)
    for offset,key in [(-.18,'rect_discrepancy'),(.18,'p2_discrepancy')]: ax.bar(x+offset,[np.mean([r[key] for r in features if r['stage']==s]) for s in ('S1','S3','Final')],.36,label=key)
    ax.set_xticks(x,['S1','S3','Final']);ax.set_yscale('log');ax.legend();fig.tight_layout();fig.savefig(folder/'normal_feature_drift.png');plt.close(fig)

def run(precheck=False):
    protocol_gate(); hashes={'Rect':hash_gate(rect.CHECKPOINT,rect.SHA),'P2':hash_gate(p2.CHECKPOINT,p2.SHA)}
    if not precheck and (OUT/'gradient_report.json').exists(): raise ValueError('FAIL_EXISTING_REPORT')
    audit=calibration.implementation_audit()
    rows,layers,meta=gradient_run(precheck)
    if precheck:
        result=dict(status='PASS',checkpoint_sha=hashes,gradient_precheck=meta,official_loss_audit=audit)
        save('precheck_report.json',result);print('PRECHECK PASS: fresh seed42; 64 normal train fixed; SHA PASS; shared='+str(meta['shared'])+'; no optimizer step; no test',flush=True);return
    parameters,features,feature_report=checkpoint_drift()
    rect.write_csv(OUT/'sample_gradient_metrics.csv',rows);rect.write_csv(OUT/'layer_gradient_metrics.csv',layers)
    rect.write_csv(OUT/'parameter_drift.csv',parameters);rect.write_csv(OUT/'normal_feature_drift.csv',features)
    stats={k:summary([r[k] for r in rows]) for k in rows[0] if isinstance(rows[0][k],(int,float))}
    counts={}
    for key in ('cos_official_s1','cos_final_s1'):
        a=np.array([r[key] for r in rows]);counts[key]=dict(negative=int(sum(a<0)),strong_conflict=int(sum(a<-.25)),aligned=int(sum(a>.25)),near_orthogonal=int(sum((a>=-.25)&(a<=.25))))
    a=np.array([r['weighted_ratio_vs_official'] for r in rows]);counts['magnitude']=dict(DOMINANT_AUX_GRADIENT=int(sum(a>=2)),COMPARABLE_HIGH=int(sum((a>=1)&(a<2))),COMPARABLE=int(sum((a>=.5)&(a<1))),LOW=int(sum(a<.5)))
    figures(rows,parameters,features)
    result=dict(status='PASS',samples=64,lambda_s1=LAMBDA,checkpoint_sha=hashes,gradient=meta,statistics=stats,counts=counts,decision=classify(rows),lambda_current_over_gradnorm=LAMBDA/stats['lambda_equal_norm']['median'],parameter_drift=parameters,normal_feature=feature_report,parameter_mutation=False,test_data_used=False,official_loss_audit=audit,limitations='Fresh-initialization gradient geometry and trained-model drift are diagnostic associations, not a causal proof. Projection is mathematical only; no update. Direction rule uses the specified OR.')
    if hash_gate(rect.CHECKPOINT,rect.SHA)!=hashes['Rect'] or hash_gate(p2.CHECKPOINT,p2.SHA)!=hashes['P2']:raise ValueError('FAIL_CHECKPOINT_MUTATION')
    save('gradient_report.json',result)
    text='========================================\nPHASE 14C-A - S1 GRADIENT INTERFERENCE AUDIT\n========================================\n'+json.dumps({k:v for k,v in result.items() if k not in ('official_loss_audit','normal_feature','gradient')},indent=2)+'\nNORMAL FEATURE\n'+json.dumps(feature_report['means'],indent=2)+'\nSHALLOW_DEEP_TRADEOFF: '+str(feature_report['shallow_deep_tradeoff'])+'\nShared gradient params: '+str(meta['shared'])+'\nShared numel: '+str(meta['shared_numel'])+'\nSTATUS: PASS\n'
    (OUT/'gradient_report.txt').write_text(text,encoding='utf-8');print(text,flush=True)
