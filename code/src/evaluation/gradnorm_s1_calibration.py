"""Phase14C-B: fixed gradient-norm coefficient, normal-only bounded calibration."""
import ast
import csv
import gc
import inspect
import json
import random
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from anomalib.engine import Engine
from src.models import multiscale_st_s1 as s1
from src.evaluation import multiscale_st_calibration as cal
from src.evaluation import gradient_interference_audit as audit
from src.evaluation.efficientad_branch_audit import parameter_hash, state_hash
from src.evaluation.intermediate_feature_audit import Capture

LAMBDA = 0.06736559
OLD_LAMBDA = 1.8656724688129778
OUT = cal.ROOT / 'experiments/phase14c_b_gradnorm_s1_calibration'
STEPS = (0, 50, 250, 500)


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def protocol():
    return dict(method='P2-GN-S1', lambda_s1=LAMBDA, old_lambda=OLD_LAMBDA,
                dynamic_lambda=False, lambda_recalibration=False, lambda_frozen=True,
                seed=42, checkpoint_finetune=False, input_hw=[256,1024],
                s1_shape=[128,253,1021], final_shape=[384,56,248], parameters=8058628,
                new_parameters=0, pcgrad=False, projection_training=False,
                dynamic_balancing=False, test_data_used=False, normal_validation_count=19,
                max_steps=500, smoke_steps=3, precision='FP32', batch=1,
                snapshot_steps=list(STEPS), snapshot_samples=16, initial_gradient_samples=64,
                optimizer='Inherited EfficientAd.configure_optimizers, unchanged',
                schedule='Inherited official scheduler, same bounded 500-step protocol as Phase14B-A',
                fusion=[0.5,0.5], quantile_probabilities=[0.9,0.995], amp=False,
                torch_compile=False, validation_model_selection=False,
                feature_probe='S1/S3/Final diagnostic only; S3 never enters loss or inference')


def protocol_gate():
    for path in (Path(__file__), cal.ROOT/'scripts/calibrate_gradnorm_s1.py'):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func,'id','')
                if name.lower() in {'test','test_dataloader','predict','auroc','aupro','load_from_checkpoint',
                                    'calibrate_lambda','loss_ratio','gradient_metrics_projection','compile','autocast'}:
                    raise ValueError('FAIL_FORBIDDEN_PATH: '+name)
    return cal.protocol_gate()


def fresh_training_model():
    return cal.fresh_training_model(LAMBDA)


@contextmanager
def preserved_rng():
    py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py); np.random.set_state(np_state); torch.set_rng_state(cpu)
        if cuda is not None: torch.cuda.set_rng_state_all(cuda)


def gradient_metrics(official, shallow):
    o, s = (v.detach().cpu().double().flatten() for v in (official, shallow))
    no, ns = (float(torch.linalg.vector_norm(v)) for v in (o,s))
    if not np.isfinite([no,ns]).all() or min(no,ns)<=0:
        raise ValueError('FAIL_ZERO_OR_NONFINITE_GRADIENT')
    resultant = o + LAMBDA*s
    c = audit.cosine(resultant,o)
    return dict(cosine=audit.cosine(o,s), raw_ratio=ns/no, weighted_ratio=LAMBDA*ns/no,
                resultant_cosine=c, resultant_angle=float(np.degrees(np.arccos(np.clip(c,-1,1)))),
                resultant_norm_ratio=float(torch.linalg.vector_norm(resultant))/no)


def training_probe_input(model,path):
    """Match Engine dataset preprocessing (CPU training transform, antialias=True)."""
    native=audit.rect.read_image(path,as_tensor=True)
    x=model.pre_processor.transform(native).unsqueeze(0).to(model.device)
    s1.input_gate(x)
    return x


def sample_gradient(model, path, extra, input_fn=cal.normal_input):
    model.zero_grad(set_to_none=True)
    x=input_fn(model,path)
    t,s,_,distance=s1.s1_features(model,x)
    official=model.model.compute_losses(x,extra.to(model.device),distance)
    # Retain Phase14C-A's exact hard-loss RNG order (including Final audit loss).
    final=s1.hard_loss(distance)
    shallow=s1.hard_loss(s1.discrepancy(t,s))
    losses=dict(official=sum(official),final=final,s1=shallow)
    if not all(bool(torch.isfinite(v)) for v in losses.values()): raise ValueError('FAIL_NONFINITE_LOSS')
    grads=audit.extract_gradients(model.model.student,losses)
    shared=sorted(set(audit.nonzero_names(grads['s1'])) & set(audit.nonzero_names(grads['official'])))
    if shared!=['conv1.bias','conv1.weight']: raise ValueError('FAIL_SHARED_PARAMETERS')
    if any(p.grad is not None for p in model.model.teacher.parameters()): raise ValueError('FAIL_TEACHER_GRAD')
    row=gradient_metrics(audit.flatten(grads['official'],shared),audit.flatten(grads['s1'],shared))
    row.update(shared=shared,**{k+'_loss':float(v.detach()) for k,v in losses.items()})
    return row


def summarize(rows):
    keys=('cosine','raw_ratio','weighted_ratio','resultant_cosine','resultant_angle','resultant_norm_ratio')
    return dict(medians={k:float(np.median([r[k] for r in rows])) for k in keys},
                strong_conflict_count=sum(r['cosine']<-.25 for r in rows), samples=len(rows), rows=rows)


def snapshot(model, paths, extras, step):
    before=parameter_hash(model)
    rows=[]
    with preserved_rng():
        # Same stochastic hard mining and augmentation for all four snapshots.
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        try:
            for path,extra in zip(paths,extras):
                row=sample_gradient(model,path,extra,input_fn=training_probe_input)
                rows.append(dict(sample_id=path.stem,**row))
        finally:
            model.zero_grad(set_to_none=True)
    after=parameter_hash(model)
    if before!=after: raise ValueError('FAIL_SNAPSHOT_PARAMETER_MUTATION')
    return dict(step=step,parameters_before=before,parameters_after=after,parameter_hash_unchanged=True,
                sample_ids=[p.stem for p in paths],rng_restored=True,
                preprocessing='Actual Engine training transform, CPU resize antialias=True',**summarize(rows))


def initial_gate(medians):
    return (.5<=medians['weighted_ratio']<=2., medians['resultant_cosine']<.5)


def stability_gate(snapshots):
    return all(str(step) in snapshots and .25<=snapshots[str(step)]['medians']['weighted_ratio']<=4.
               and snapshots[str(step)]['medians']['resultant_cosine']>=.25 for step in STEPS[1:])


def initial_gradient():
    model,meta=audit.fresh_audit_model()
    model.freeze_lambda(LAMBDA)
    reference=json.loads((audit.OUT/'gradient_report.json').read_text(encoding='utf-8'))
    paths=audit.select_samples(cal.normal_paths('train'),'train',[0]*137)
    if [p.stem for p in paths]!=reference['gradient']['sample_ids']: raise ValueError('FAIL_SAMPLE_IDS')
    before=parameter_hash(model); rows=[]; iterator=iter(model.imagenet_loader)
    for i,path in enumerate(paths):
        try: extra=next(iterator)[0]
        except StopIteration:
            iterator=iter(model.imagenet_loader); extra=next(iterator)[0]
        rows.append(dict(sample_id=path.stem,**sample_gradient(model,path,extra)))
        print(f'INITIAL GRADIENT {i+1}/64: weighted={rows[-1]["weighted_ratio"]:.6f}',flush=True)
    model.zero_grad(set_to_none=True)
    result=dict(initialization=meta,sample_ids=[p.stem for p in paths],same_phase14ca_ids=True,
                parameters_before=before,parameters_after=parameter_hash(model),**summarize(rows))
    if before!=result['parameters_after']: raise ValueError('FAIL_INITIAL_PARAMETER_MUTATION')
    passed,caution=initial_gate(result['medians'])
    result.update(gradient_balance_gate='PASS' if passed else 'FAIL_GRADIENT_BALANCE',
                  DIRECTION_PRESERVATION_CAUTION=caution)
    save('initial_gradient.json',result)
    del model; gc.collect(); torch.cuda.empty_cache()
    return result


@torch.no_grad()
def feature_values(model):
    training=model.training; model.eval()
    before=parameter_hash(model); rows=[]
    wanted={k:['conv1.post_relu','conv2.post_relu','conv4'] for k in ('teacher','student')}
    try:
        for path in cal.normal_paths('validation'):
            x=training_probe_input(model,path)
            with Capture(model,wanted) as capture: model.model.compute_student_teacher_distance(x)
            row=dict(sample_id=path.stem)
            for stage,name in [('S1','conv1.post_relu'),('S3','conv2.post_relu'),('Final','conv4')]:
                t,s=capture.values['teacher',name],capture.values['student',name]
                if stage=='Final':
                    s=s[:,:model.model.teacher_out_channels]
                    t=(t-model.model.mean_std['mean'])/model.model.mean_std['std']
                row[stage]=float((t-s).square().mean())
            if not np.isfinite([row[k] for k in ('S1','S3','Final')]).all(): raise ValueError('FAIL_FEATURE_NUMERICS')
            rows.append(row)
    finally: model.train(training)
    if before!=parameter_hash(model): raise ValueError('FAIL_FEATURE_PARAMETER_MUTATION')
    return dict(samples=19,rows=rows,means={k:float(np.mean([r[k] for r in rows])) for k in ('S1','S3','Final')})


def feature_gate(means):
    return means['S1']['trained']<means['S1']['initial'] and (
        means['Final']['trained']-means['Final']['initial'])/(means['Final']['initial']+1e-12)<=.10


def feature_comparison(initial,trained):
    if [r['sample_id'] for r in initial['rows']]!=[r['sample_id'] for r in trained['rows']]:
        raise ValueError('FAIL_FEATURE_SAMPLE_IDS')
    means={}
    for key in ('S1','S3','Final'):
        a,b=initial['means'][key],trained['means'][key]
        means[key]=dict(initial=a,trained=b,change=b-a,relative_change=(b-a)/(a+1e-12),
                        discrepancy_reduction=(a-b)/(a+1e-12))
    return dict(status='PASS',source='19 normal validation',initial=initial,trained=trained,means=means,
                health_gate=feature_gate(means),test_data_used=False,model_selection=False,
                final_degradation_ratio=means['Final']['relative_change'])


def update_gate(report):
    for key in ('teacher_unchanged','student_updated','student_conv1_updated','student_deeper_updated','ae_updated','lambda_unchanged'):
        if not report[key]: raise ValueError('FAIL_UPDATE_'+key)


def steady_wall_seconds(start,end,diagnostic_seconds):
    return (end-start-diagnostic_seconds)/450


class Monitor(cal.CalibrationMonitor):
    def __init__(self,steps):
        super().__init__(f'steps{steps}',steps,LAMBDA)
        self.snapshots={}; self.initial_features=None; self.step_seconds=[]
        self.diagnostic_seconds=0.

    def on_train_batch_start(self,trainer,pl_module,batch,batch_idx):
        if self.target==500 and not self.snapshots:
            # This hook runs after the official module's on_train_start statistics.
            with preserved_rng():
                self.paths=cal.normal_paths('train')[:16]
                iterator=iter(pl_module.imagenet_loader)
                self.extras=[next(iterator)[0].cpu().clone() for _ in self.paths]
                self.initial_features=feature_values(pl_module)
                save('normal_feature_probe.json',dict(status='INITIAL_ONLY',initial=self.initial_features))
            self.take_snapshot(pl_module,0)
        torch.cuda.synchronize(); self.batch_start=time.perf_counter()

    def take_snapshot(self,model,step):
        torch.cuda.synchronize(); started=time.perf_counter()
        self.snapshots[str(step)]=snapshot(model,self.paths,self.extras,step)
        save('gradient_snapshots.json',self.snapshots)
        print(f'SNAPSHOT {step}: '+json.dumps(self.snapshots[str(step)]['medians']),flush=True)
        torch.cuda.synchronize(); self.diagnostic_seconds+=time.perf_counter()-started

    def on_train_batch_end(self,trainer,pl_module,outputs,batch,batch_idx):
        torch.cuda.synchronize(); self.step_seconds.append(time.perf_counter()-self.batch_start)
        step=int(trainer.global_step)
        if float(pl_module.lambda_s1)!=LAMBDA or step!=len(self.rows)+1 or step>self.target:
            raise ValueError('FAIL_TRAINING_STATE')
        s1.input_gate(batch.image)
        row=dict(phase=self.phase,step=step,epoch=int(trainer.current_epoch),
                 learning_rate=trainer.optimizers[0].param_groups[0]['lr'],**pl_module.last_loss_log)
        if not np.isfinite([v for k,v in row.items() if k!='phase']).all(): raise ValueError('FAIL_NUMERICS')
        self.rows.append(row)
        with (OUT/'training_log.csv').open('a',encoding='utf-8',newline='') as stream:
            csv.DictWriter(stream,fieldnames=cal.LOG_FIELDS).writerow(row)
        if step==50:
            self.warm_end=time.perf_counter(); self.warm_diagnostic_seconds=self.diagnostic_seconds
        if step==500:
            self.end=time.perf_counter()
            self.steady_wall=steady_wall_seconds(self.warm_end,self.end,self.diagnostic_seconds-self.warm_diagnostic_seconds)
        if step<=3 or step%25==0:
            print(f'{self.phase} {step}/{self.target}: total={row["p2_total"]:.6f}, final={row["final_st"]:.6f}, S1={row["s1_raw"]:.6f}',flush=True)
        if self.target==500 and step in STEPS[1:]: self.take_snapshot(pl_module,step)


def train_run(steps):
    options=cal.trainer_options(steps)
    model,initialization=fresh_training_model()
    dm,_=cal.base.dataset()
    branches=('teacher','student','ae')
    before={k:parameter_hash(getattr(model.model,k)) for k in branches}
    conv_before={k:parameter_hash(getattr(model.model.student,k)) for k in ('conv1','conv2','conv3','conv4')}
    quantiles={k:v.detach().cpu().clone() for k,v in model.model.quantiles.items()}
    monitor=Monitor(steps)
    engine=Engine(**options,default_root_dir=str(OUT),callbacks=[monitor])
    report=dict(status='RUNNING',completed_steps=0,target_steps=steps,NaN=False,Inf=False,OOM=False,
                initialization=initialization,lambda_s1=LAMBDA,validation_batches=0)
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    try:
        engine.fit(model=model,datamodule=dm)
        report['total_fit_seconds']=time.perf_counter()-started
        report['completed_steps']=int(engine.trainer.global_step)
        if report['completed_steps']!=steps or len(monitor.rows)!=steps: raise ValueError('FAIL_STEP_COUNT')
        after={k:parameter_hash(getattr(model.model,k)) for k in branches}
        conv_after={k:parameter_hash(getattr(model.model.student,k)) for k in conv_before}
        report.update(branch_hashes_before=before,branch_hashes_after=after,
                      conv_hashes_before=conv_before,conv_hashes_after=conv_after,
                      teacher_unchanged=before['teacher']==after['teacher'],
                      student_updated=before['student']!=after['student'],ae_updated=before['ae']!=after['ae'],
                      student_conv1_updated=conv_before['conv1']!=conv_after['conv1'],
                      student_deeper_updated=all(conv_before[k]!=conv_after[k] for k in ('conv2','conv3','conv4')),
                      lambda_unchanged=float(model.lambda_s1)==LAMBDA,
                      parameters=sum(p.numel() for p in model.parameters()),last_losses=monitor.rows[-1])
        update_gate(report)
        if report['parameters']!=8058628: raise ValueError('FAIL_PARAMETERS')
        if any(not torch.equal(v,model.model.quantiles[k].cpu()) for k,v in quantiles.items()) or bool(model.s1_quantiles_fitted):
            raise ValueError('FAIL_TRAINING_QUANTILE_MUTATION')
        # Capture training/snapshot memory before inference-only probes.
        report.update(peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                      peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
        if report['peak_allocated_mib']>6500: raise ValueError('FAIL_MEMORY_LIMIT')
        if steps==500:
            steady=monitor.steady_wall
            report.update(steady_seconds_per_step=steady,projected_70000_hours=steady*70000/3600,
                          timing_scope='Wall time between steps50 and500, including data loading/transfer; subtract measured snapshots50/250 only',
                          batch_only_seconds_per_step=float(np.mean(monitor.step_seconds[50:])),
                          diagnostics_seconds=monitor.diagnostic_seconds,
                          step_seconds=monitor.step_seconds,GRADIENT_BALANCE_STABLE=stability_gate(monitor.snapshots))
            fields=('final_st','s1_raw','weighted_s1','official_total','p2_total','aux_fraction')
            for tag,rows in [('first50',monitor.rows[:50]),('last50',monitor.rows[-50:])]:
                report[tag+'_medians']={k:float(np.median([r[k] for r in rows])) for k in fields}
            with preserved_rng():
                feature=feature_comparison(monitor.initial_features,feature_values(model))
            save('normal_feature_probe.json',feature)
            report['normal_feature']=feature
            report['runtime']=runtime_probe(model)
        report['status']='PASS'
    except Exception as exc:
        report.update(status='FAIL',error=str(exc),completed_steps=len(monitor.rows),
                      OOM=isinstance(exc,torch.cuda.OutOfMemoryError))
        if 'NUMERIC' in str(exc) or 'NONFINITE' in str(exc):
            report.update(NaN=any(bool(torch.isnan(p).any()) for p in model.parameters()),
                          Inf=any(bool(torch.isinf(p).any()) for p in model.parameters()),numerical_failure=True)
        raise
    finally:
        report.setdefault('peak_allocated_mib',torch.cuda.max_memory_allocated()/1024**2)
        report.setdefault('peak_reserved_mib',torch.cuda.max_memory_reserved()/1024**2)
        save(f'training_{steps}_report.json',report)
        del engine,model; gc.collect(); torch.cuda.empty_cache()
    return report


@torch.no_grad()
def runtime_probe(model):
    # Lightning teardown may move the fitted module back to CPU.
    model.to('cuda:0')
    model.eval(); before=parameter_hash(model)
    # Official quantile probabilities; local values only, no selection or training.
    maps=[]
    for path in cal.normal_paths('validation'):
        t,s,_,_=s1.s1_features(model,cal.normal_input(model,path))
        maps.append(s1.s1_map(s1.discrepancy(t,s)))
    qa,qb=model._get_quantiles_of_maps(maps)
    del maps,t,s
    x=cal.normal_input(model,cal.normal_paths('train')[0])
    for _ in range(20): output=s1.inference_outputs(model,x,qa,qb)
    del output
    timings=[]
    for _ in range(100):
        torch.cuda.synchronize(); start=time.perf_counter()
        output=s1.inference_outputs(model,x,qa,qb)
        torch.cuda.synchronize(); timings.append((time.perf_counter()-start)*1000)
        if output['NEW_MODEL_MS_ST_ONLY'].shape!=(1,1,256,1024): raise ValueError('FAIL_RUNTIME_SHAPE')
        del output
    if before!=parameter_hash(model): raise ValueError('FAIL_RUNTIME_MUTATION')
    result=dict(status='PASS',warmup=20,runs=100,mean_ms=float(np.mean(timings)),
                median_ms=float(np.median(timings)),samples_ms=timings,parameters_unchanged=True,
                scope='Unchanged S1+Final inference_outputs; local normal S1 quantiles, no final quantile fitting or performance evaluation',
                test_data_used=False)
    save('runtime_probe.json',result)
    return result


def precheck():
    protocol_gate(); implementation=cal.implementation_audit()
    model,meta=fresh_training_model()
    other,other_meta=fresh_training_model()
    if parameter_hash(model)!=parameter_hash(other): raise ValueError('FAIL_FRESH_SEED42')
    del other; gc.collect(); torch.cuda.empty_cache()
    model.prepare_imagenette_data(s1.TARGET)
    extra=next(iter(model.imagenet_loader))[0]
    s1.input_gate(extra)
    model.eval()
    with torch.no_grad():
        t,s,_,d=s1.s1_features(model,cal.normal_input(model,cal.normal_paths('train')[0]))
    result=dict(status='PASS',protocol=protocol(),initialization=meta,seed_reproduction=True,
                input=[1,3,256,1024],s1=list(s.shape),final=list(d.shape),
                parameters=sum(p.numel() for p in model.parameters()),lambda_s1=float(model.lambda_s1),
                lambda_frozen=bool(model.lambda_frozen),imagenette_shape=list(extra.shape),official_loss_audit=implementation)
    save('precheck_report.json',result)
    del model,t,s,d; gc.collect(); torch.cuda.empty_cache()
    print('PRECHECK PASS',flush=True)
    return result


def old_comparison():
    path=cal.OUT/'calibration_report.json'
    if not path.exists(): return dict(available=False,reason='No old report')
    old=json.loads(path.read_text(encoding='utf-8'))
    run=old.get('calibration',{})
    keys=('first50_medians','last50_medians','steady_seconds_per_step','projected_70000_hours','peak_allocated_mib','peak_reserved_mib')
    return dict(available=True,source=str(path),descriptive_only=True,rerun=False,
                calibration={k:run[k] for k in keys if k in run},runtime=old.get('runtime',{}))


def finalize_decision(report):
    report['snapshots']=json.loads((OUT/'gradient_snapshots.json').read_text(encoding='utf-8'))
    report['old_p2_comparison']=old_comparison()
    run_report=report['calibration']
    feasible=(run_report['completed_steps']==500 and not any(run_report[k] for k in ('NaN','Inf','OOM'))
              and run_report['GRADIENT_BALANCE_STABLE'] and run_report['teacher_unchanged']
              and run_report['parameters']==8058628 and run_report['peak_allocated_mib']<=6500
              and report['initial_gradient']['gradient_balance_gate']=='PASS')
    report['FORMAL_GN_TRAINING_FEASIBLE']=feasible
    report['PROCEED_TO_GN_70K']=feasible and run_report['normal_feature']['health_gate']
    report['status']='PASS' if report['PROCEED_TO_GN_70K'] else 'FAIL'
    if not report['PROCEED_TO_GN_70K']:
        report['error']='Calibration completed; gradient stability or normal feature health gate failed'


def recovery_gate(report):
    run_report=report.get('calibration',{})
    if run_report.get('completed_steps')!=500 or 'Input type (torch.cuda.FloatTensor)' not in report.get('error',''):
        raise ValueError('FAIL_RUNTIME_RECOVERY_PROVENANCE')
    update_gate(run_report)
    if any(run_report[k] for k in ('NaN','Inf','OOM')): raise ValueError('FAIL_RUNTIME_RECOVERY_NUMERICS')


def recover_runtime_only():
    """Finish structure-only timing after a completed500 device failure; no training."""
    import shutil
    report=json.loads((OUT/'calibration_report.json').read_text(encoding='utf-8'))
    recovery_gate(report)
    archive=OUT/'attempts/completed500_runtime_device_failure'
    archive.mkdir(parents=True,exist_ok=False)
    for name in ('calibration_report.json','calibration_report.txt','training_500_report.json','runtime_probe.json'):
        shutil.copyfile(OUT/name,archive/name)
    model,meta=fresh_training_model()
    result=runtime_probe(model)
    result.update(model_scope='Fresh same-architecture structural reference, no trained500 checkpoint saved',
                  recovery='Runtime only; zero optimizer steps; completed500 scientific records retained')
    save('runtime_probe.json',result)
    del model; gc.collect(); torch.cuda.empty_cache()
    report['calibration']['runtime']=result
    report['calibration']['status']='PASS'
    report['calibration'].pop('error',None)
    report.pop('traceback',None); report.pop('error',None)
    report['implementation_recovery']=dict(reason='Engine teardown moved model to CPU; explicit CUDA restore added',
                                          training_rerun=False,archive=str(archive),runtime_reference_initialization=meta)
    save('training_500_report.json',report['calibration'])
    finalize_decision(report)
    save('calibration_report.json',report)
    text=render(report); (OUT/'calibration_report.txt').write_text(text,encoding='utf-8'); print(text,flush=True)
    return report


def render(report):
    initial=report.get('initial_gradient',{}); med=initial.get('medians',{})
    smoke=report.get('smoke',{}); run=report.get('calibration',{})
    lines=['='*40,'PHASE 14C-B - GRADNORM S1 CALIBRATION','='*40,
           'Method: P2-GN-S1','Input: 256x1024','S1: 128x253x1021',
           f'Old lambda: {OLD_LAMBDA}',f'New lambda: {LAMBDA}','Parameters: 8058628','New parameters: 0',
           '-'*40,'INITIAL GRADIENT','Weighted S1 / Official:','Old median: 27.6961',
           'New median: '+str(med.get('weighted_ratio','NOT_RUN')),
           'Official/S1 cosine: '+str(med.get('cosine','NOT_RUN')),
           'GN resultant cosine vs Official: '+str(med.get('resultant_cosine','NOT_RUN')),
           'GN resultant angle: '+str(med.get('resultant_angle','NOT_RUN')),
           'Gradient balance gate: '+initial.get('gradient_balance_gate','NOT_RUN'),
           '-'*40,'3-STEP','Completed: '+str(smoke.get('completed_steps','NOT_RUN'))]
    for label,key in [('Official total','official_total'),('Final ST','final_st'),('S1 raw','s1_raw'),('Weighted S1','weighted_s1'),('AE','ae'),('STAE','stae'),('Total','p2_total')]:
        lines.append(label+': '+str(smoke.get('last_losses',{}).get(key,'NOT_RUN')))
    for key in ('teacher_unchanged','student_conv1_updated','student_deeper_updated','ae_updated','NaN','Inf','OOM'):
        lines.append(key+': '+str(smoke.get(key,'NOT_RUN')))
    lines+=['-'*40,'500-STEP']
    for label,key in [('Completed','completed_steps'),('Seconds/step','steady_seconds_per_step'),('Projected 70k (h)','projected_70000_hours'),('Peak allocated (MiB)','peak_allocated_mib'),('Peak reserved (MiB)','peak_reserved_mib'),('Teacher unchanged','teacher_unchanged'),('NaN','NaN'),('Inf','Inf'),('OOM','OOM')]:
        lines.append(label+': '+str(run.get(key,'NOT_RUN')))
    lines+=['-'*40,'GRADIENT SNAPSHOTS']
    for step in STEPS:
        lines.append('Step '+str(step)+': '+json.dumps(report.get('snapshots',{}).get(str(step),{}).get('medians','NOT_RUN')))
    lines.append('GRADIENT_BALANCE_STABLE: '+str(run.get('GRADIENT_BALANCE_STABLE',False)))
    lines+=['-'*40,'NORMAL FEATURE MATCHING']
    for stage in ('S1','Final'):
        values=run.get('normal_feature',{}).get('means',{}).get(stage,{})
        lines.extend([f'Initial {stage} discrepancy: '+str(values.get('initial','NOT_RUN')),
                      f'500-step {stage}: '+str(values.get('trained','NOT_RUN')),
                      'Change: '+str(values.get('relative_change','NOT_RUN'))])
    lines+=['-'*40,'DECISION','FORMAL_GN_TRAINING_FEASIBLE: '+str(report.get('FORMAL_GN_TRAINING_FEASIBLE',False)),
            'PROCEED_TO_GN_70K: '+str(report.get('PROCEED_TO_GN_70K',False)),
            'Test data used: False','STATUS: '+report['status']]
    if report.get('error'): lines.append('Error: '+report['error'])
    return '\n'.join(lines+['='*40])


def run():
    if (OUT/'calibration_report.json').exists(): raise ValueError('FAIL_EXISTING_REPORT')
    if json.loads((OUT/'precheck_report.json').read_text(encoding='utf-8'))['status']!='PASS':
        raise ValueError('FAIL_PRECHECK_REQUIRED')
    save('config.json',protocol())
    for name in ('gradient_snapshots.json','normal_feature_probe.json','runtime_probe.json'):
        save(name,dict(status='NOT_RUN'))
    with (OUT/'training_log.csv').open('x',encoding='utf-8',newline='') as stream:
        csv.DictWriter(stream,fieldnames=cal.LOG_FIELDS).writeheader()
    report=dict(status='RUNNING',FORMAL_GN_TRAINING_FEASIBLE=False,PROCEED_TO_GN_70K=False,test_data_used=False)
    try:
        protocol_gate()
        report['initial_gradient']=initial_gradient()
        if report['initial_gradient']['gradient_balance_gate']!='PASS': raise ValueError('FAIL_GRADIENT_BALANCE')
        report['smoke']=train_run(3)
        report['calibration']=train_run(500)
        finalize_decision(report)
    except Exception as exc:
        report.update(status='FAIL',error=str(exc),traceback=traceback.format_exc())
        for key,steps in [('smoke',3),('calibration',500)]:
            path=OUT/f'training_{steps}_report.json'
            if path.exists(): report[key]=json.loads(path.read_text(encoding='utf-8'))
    finally:
        save('calibration_report.json',report)
        text=render(report); (OUT/'calibration_report.txt').write_text(text,encoding='utf-8')
        print(text,flush=True)
    return report
