"""Frozen Phase14B-C inference, reference checks and synchronized benchmarks."""
import ast
import gc
import inspect
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from src.models.multiscale_st_s1 import (MultiScaleSTS1, s1_features, discrepancy,
    s1_map, normalized_map, fuse, input_gate, TARGET)
from src.evaluation import rectangular_formal_evaluation as rect
from src.evaluation.efficientad_branch_audit import parameter_hash, state_hash
from src.evaluation.st_feature_rootcause import map_from_feature, padding_protocol

ROOT=rect.ROOT
OUT=ROOT/'experiments/phase14b_c_s1_public'
BASE=rect.OUT
CHECKPOINT=ROOT/'experiments/phase14b_b_s1_formal_seed42/checkpoints/final.ckpt'
SHA='24645c5d1d30caf2754769ab60c94681b431b1c6f4337eb52eb1d631f5dffa7b'
LAMBDA=1.8656724688129778
MODES=('NEW_MODEL_FINAL_ONLY','NEW_MODEL_MS_ST_ONLY','NEW_MODEL_MS_COMBINED')
SLUGS=dict(zip(MODES,('final_only','ms_st','ms_combined')))
AUX=('NEW_FINAL_ST','S1')
NATIVE=rect.NATIVE
KEYS=rect.METRIC_KEYS
GROUP_COUNTS=rect.GROUP_COUNTS
exact=rect.exact
metric_engine=rect.metrics
native_map=rect.native_map
input_tensor=rect.input_tensor
timed=rect.timed
read_json=rect.read_json
read_csv=rect.read_csv
write_json=rect.write_json
write_csv=rect.write_csv
digest=rect.digest
EPS=1e-10
RECT=dict(zip(KEYS,np.array([87.9167,84.0931,19.6522,44.8072])/100))


def checkpoint_gate(path=CHECKPOINT):
    if digest(path)!=SHA:raise ValueError('FAIL_CHECKPOINT_HASH')
    return SHA


def load_model():
    import anomalib
    if anomalib.__version__!='2.6.1' or not torch.cuda.is_available():raise ValueError('FAIL_ENVIRONMENT')
    checkpoint_gate()
    report=read_json(CHECKPOINT.parent.parent/'training_report.json')
    saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    if report['status']!='PASS' or saved['global_step']!=70000 or not saved['phase14bb']['final_validation_completed']:
        raise ValueError('FAIL_CHECKPOINT_PROVENANCE')
    del saved
    model=MultiScaleSTS1.load_from_checkpoint(str(CHECKPOINT),strict=True,map_location='cpu',weights_only=False)
    if sum(p.numel() for p in model.parameters())!=8058628:raise ValueError('FAIL_PARAMETERS')
    if float(model.lambda_s1)!=LAMBDA or not bool(model.lambda_frozen):raise ValueError('FAIL_LAMBDA_STATE')
    q={k:float(v.detach()) for k,v in model.model.quantiles.items()}
    q.update(qa_s1=float(model.qa_s1),qb_s1=float(model.qb_s1))
    if q!=report['final_quantiles'] or not bool(model.s1_quantiles_fitted):raise ValueError('FAIL_QUANTILE_STATE')
    if not all(np.isfinite(list(q.values()))) or any(q['qb_'+b]<=q['qa_'+b] for b in ('st','ae','s1')):
        raise ValueError('FAIL_QUANTILE_STATE')
    if not model.model.is_set(model.model.mean_std):raise ValueError('FAIL_TEACHER_STATISTICS')
    if repr(model.pre_processor.transform)!=repr(rect.preprocessor().transform):raise ValueError('FAIL_PREPROCESSOR')
    if model.model.ae.decoder.padding is not False or model.model.pad_maps is not True:raise ValueError('FAIL_PADDING')
    model.eval().to('cuda:0')
    # Inspect the actual installed methods, not a paper/version assumption.
    source={k:dict(signature=str(inspect.signature(getattr(type(model.model),k))),source=inspect.getsource(getattr(type(model.model),k)))
            for k in ('forward','compute_maps','compute_student_teacher_distance')}
    return model,dict(checkpoint_sha256=SHA,strict_load=True,parameters=8058628,new_parameters=0,quantiles=q,
        lambda_s1=LAMBDA,input_size_hw=list(TARGET),preprocessor=repr(model.pre_processor.transform),official_source=source,
        formulas=dict(final_only='0.5*Final_ST_norm + 0.5*AE_norm',ms_st='0.5*S1_norm + 0.5*Final_ST_norm',
                      ms_combined='0.5*MS_ST + 0.5*AE_norm'),padding=list(padding_protocol(model)))


def final_map(model,distance):
    raw=map_from_feature(distance.mean(1,keepdim=True),padding_protocol(model),TARGET)
    q=model.model.quantiles
    return normalized_map(raw,q['qa_st'],q['qb_st'])


@torch.no_grad()
def forward_mode(model,x,mode):
    input_gate(x)
    if model.training or model.model.training:raise ValueError('FAIL_EVAL_MODE')
    if mode==MODES[0]:return model.model(x).anomaly_map
    if mode not in MODES:raise ValueError('FAIL_OUTPUT_MODE')
    t,s,student,distance=s1_features(model,x)
    shallow=normalized_map(s1_map(discrepancy(t,s)),model.qa_s1,model.qb_s1)
    shallow=F.interpolate(shallow,size=TARGET,mode='bilinear',align_corners=False)
    if mode==MODES[1]:
        return 0.5*shallow+0.5*final_map(model,distance)
    final,ae=model.model.compute_maps(x,student,distance,normalize=True)
    return fuse(shallow,final,ae)[mode]


@torch.no_grad()
def extract(model,x):
    input_gate(x)
    if model.training or model.model.training:raise ValueError('FAIL_EVAL_MODE')
    t,s,student,distance=s1_features(model,x)
    final,ae=model.model.compute_maps(x,student,distance,normalize=True)
    shallow=normalized_map(s1_map(discrepancy(t,s)),model.qa_s1,model.qb_s1)
    outputs=fuse(shallow,final,ae)
    outputs.update(NEW_FINAL_ST=final,S1=F.interpolate(shallow,size=TARGET,mode='bilinear',align_corners=False))
    official=model.model(x).anomaly_map
    errors=dict(final_only=float((outputs[MODES[0]]-official).abs().max()),
                final_st=float((final-final_map(model,distance)).abs().max()),
                ms_st=float((outputs[MODES[1]]-(.5*outputs['S1']+.5*final)).abs().max()),
                ms_combined=float((outputs[MODES[2]]-(.5*outputs[MODES[1]]+.5*ae)).abs().max()))
    if any(not np.isfinite(v) or v>1e-5 for v in errors.values()):raise ValueError('FAIL_RECONSTRUCTION: '+str(errors))
    if any(tuple(v.shape)!=(1,1,*TARGET) or not torch.isfinite(v).all() for v in outputs.values()):raise ValueError('FAIL_MAP_SHAPE')
    return outputs,errors,dict(teacher_s1=list(t.shape),student_s1=list(s.shape),final_st=list(distance.shape),student=list(student.shape))


def pairing_gate(rows):
    sets=[{r['sample_id'] for r in rows[k]} for k in MODES]
    if any(len(rows[k])!=114 for k in MODES) or any(len(s)!=114 or s!=sets[0] for s in sets):raise ValueError('FAIL_PAIRED_SOURCE_IDS')


def baseline_tables():
    metrics={k:read_json(BASE/(k.lower()+'_metrics.json')) for k in ('COMBINED','ST')}
    samples={(r['branch'],r['sample_id']):r for r in read_csv(BASE/'sample_metrics.csv')}
    regions={(r['branch'],r['sample_id'],int(r['region_id'])):r for r in read_csv(BASE/'region_metrics.csv')}
    if len(samples)!=180 or len(regions)!=900:raise ValueError('FAIL_BASELINE_REFERENCE')
    for k in KEYS:
        if abs(metrics['COMBINED'][k]-RECT[k])>1e-6:raise ValueError('FAIL_BASELINE_METRICS')
    return dict(metrics=metrics,samples=samples,regions=regions)


def references():
    ref=rect.references();baseline=baseline_tables();ref['baseline']=baseline
    predictions=rect.mapping_gate(read_csv(BASE/'predictions.csv'));ref['rect']=predictions
    if set(predictions)!=set(ref['mapping']):raise ValueError('FAIL_REFERENCE_MAPPING')
    files={BASE/f for f in ('combined_metrics.json','st_metrics.json','sample_metrics.csv','region_metrics.csv','predictions.csv','evaluation_report.json')}
    files.update((CHECKPOINT,CHECKPOINT.parent.parent/'training_report.json',ROOT/'src/models/multiscale_st_s1.py'))
    for sid,row in predictions.items():
        old=ref['mapping'][sid]
        if Path(row['image_path']).resolve()!=Path(old['image_path']).resolve() or int(row['gt_label'])!=int(old['gt_label']):raise ValueError('FAIL_REFERENCE_MAPPING')
        for key in ('combined_path','st_path'):
            p=Path(row[key]);a=np.load(p,mmap_mode='r',allow_pickle=False)
            if a.shape!=NATIVE or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('FAIL_BASELINE_MAP')
            files.add(p)
        if sid in ref['membership']:
            for branch in ('COMBINED','ST'):
                if (branch,sid) not in baseline['samples']:raise ValueError('FAIL_PAIRED_MEMBERS')
    for sid in ref['tiny']:
        if sum(k[0]=='ST' and k[1]==sid for k in baseline['regions'])!=1:raise ValueError('FAIL_TINY_REGION_MAPPING')
    ref['gallery_sets'].pop('B0_worst')
    ref['gallery_sets']['Rect_worst']=sorted(ref['membership'],key=lambda sid:(float(baseline['samples']['COMBINED',sid]['per_image_AUPRO005']),sid))[:10]
    ref['gallery_ids']=sorted(set(sum(ref['gallery_sets'].values(),[])))
    ref['integrity'].update({str(p):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,sha256=digest(p)) for p in sorted(files)})
    return ref


def no_training_paths():
    forbidden={'fit','train','backward','configure_optimizers','optimizer_step','validate','fit_s1_maps','freeze_lambda','on_validation_start'}
    for path in (Path(__file__),ROOT/'scripts/evaluate_multiscale_st_s1_public.py'):
        tree=ast.parse(path.read_text(encoding='utf-8'))
        for n in ast.walk(tree):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in forbidden:raise ValueError('FAIL_TRAINING_PATH')
    return True


@torch.no_grad()
def benchmark(model,ref):
    paths=[ref['mapping'][sid]['image_path'] for sid in sorted(ref['mapping'])]
    tensors=[input_tensor(model,p).cpu() for p in paths]
    result=dict(protocol=rect.runtime_protocol(),loaded_parameters=8058628,new_parameters=0,modes={},
        timing_definition='CUDA synchronized wall-clock; model-only excludes H2D, disk, preprocess and native resize. End-to-end includes disk, preprocess, H2D, necessary branches, D2H and CPU bilinear native resize.',
        reference_note='Rect and P1 latency comparisons use frozen historical benchmarks, not same-process baseline reruns.',
        memory_note='Full model, including AE weights, remains loaded in all three modes; MS-ST skips AE execution.')
    for mode in MODES:
        for i in range(20):forward_mode(model,tensors[i].to('cuda:0'),mode)
        torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        times=[];rounds=[]
        for rnd in range(5):
            local=[]
            for tensor in tensors:
                x=tensor.to('cuda:0');value,ms=timed(lambda:forward_mode(model,x,mode));local.append(ms);del value,x
            times.extend(local);rounds.append(rect.stats(local))
            print(f'{mode} model-only {rnd+1}/5: {np.mean(local):.4f} ms/source',flush=True)
        memory=dict(peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        def pipeline(path):return native_map(forward_mode(model,input_tensor(model,path),mode))
        ends=[];endrounds=[]
        for rnd in range(3):
            local=rect.benchmark_loop(pipeline,paths,rounds=1,warmup=0);ends.extend(local);endrounds.append(rect.stats(local))
            print(f'{mode} end-to-end {rnd+1}/3: {np.mean(local):.4f} ms/source',flush=True)
        endmemory=dict(peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
        result['modes'][mode]=dict(model_only=rect.stats(times),end_to_end=rect.stats(ends),model_rounds=rounds,end_to_end_rounds=endrounds,
            model_only_samples_ms=times,end_to_end_samples_ms=ends,model_memory=memory,end_to_end_memory=endmemory)
    result['gpu_total_mib']=torch.cuda.get_device_properties(0).total_memory/1024**2
    return result


def paired(new,old):
    d=np.asarray(new,dtype=float)-np.asarray(old,dtype=float)
    return dict(improved=int((d>EPS).sum()),worsened=int((d<-EPS).sum()),unchanged=int((abs(d)<=EPS).sum()),
                mean_delta_pp=float(d.mean()*100),median_delta_pp=float(np.median(d)*100))


def background_values(scores,mask):
    if scores.shape!=mask.shape:raise ValueError('FAIL_GT_SHAPE')
    return scores[~np.asarray(mask,dtype=bool)]


def continuation_gate(report):
    if not report.get('parameters_unchanged') or not report.get('state_unchanged') or report.get('precheck')!='PASS':
        raise ValueError('FAIL_CONTINUATION_INTEGRITY')
    errors=report.get('reconstruction_errors',{})
    if not errors or any(not np.isfinite(v) or v>1e-5 for v in errors.values()):raise ValueError('FAIL_RECONSTRUCTION')


def classify(metric,latency,tiny):
    gain=100*(metric['aupro005']-RECT['aupro005']);image_delta=100*(metric['image_auroc']-RECT['image_auroc'])
    # Resolve qualitative wording before observing results. Clearly improved uses
    # the same fixed +1.5pp localization requirement as MODERATE. Pareto means
    # no worse on all four metrics and latency, with at least one strict gain.
    pareto=all(metric[k]>=RECT[k] for k in KEYS) and latency<=107.6683 and (any(metric[k]>RECT[k] for k in KEYS) or latency<107.6683)
    flags=dict(P2_STRONG_SUCCESS=metric['aupro005']>=.220758 and metric['image_auroc']>=.85 and latency<=130 and tiny>=.081571,
        P2_MODERATE_SUCCESS=gain>=1.5 and image_delta>=-2 and latency/107.6683<=1.15,
        P2_LOCALIZATION_ONLY=gain>=1.5 and image_delta<-2,
        P2_NO_BENEFIT=gain<.5 and not pareto)
    flags={k:bool(v) for k,v in flags.items()}
    return dict(**flags,classification=next((k for k,v in flags.items() if v),'UNCLASSIFIED_BY_PREREGISTERED_RULES'),
        qualitative_rule_operationalization='Clearly improved means >=1.5pp AU-PRO005; other clear Pareto gain requires all four metrics and latency no worse, one strictly better. Uncovered cases remain unclassified.',
        delta_vs_rect_pp={k:float(100*(metric[k]-RECT[k])) for k in KEYS},delta_vs_p1384_pp={k:float(100*(metric[k]-rect.P1384[k])) for k in KEYS},
        latency_overhead_vs_rect_percent=float((latency/107.6683-1)*100),speedup_vs_p1384=float(196.563/latency),
        reaches_p1384_strict_and_faster=bool(metric['aupro005']>=.220758 and latency<196.563))
