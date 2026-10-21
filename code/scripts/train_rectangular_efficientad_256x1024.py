"""Phase13B: fixed rectangular formal training; public data metadata only.

--precheck-only checks gates without training. --resume restores this run's last.ckpt.
"""
import argparse
import ast
import gc
import hashlib
import json
import math
import os
import pickle
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import torchvision
from anomalib.callbacks import ModelCheckpoint
from anomalib.engine import Engine
from anomalib.models import EfficientAd
from anomalib.data.utils import read_image
from lightning.pytorch.callbacks import LearningRateMonitor,EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from src.evaluation import rectangular_efficientad_calibration as cal
from scripts.train_baseline384 import BaselineMonitor as VerifiedMonitor
from scripts.train_p1_halo96 import check_logs

OUT=ROOT/'experiments/phase13b_rectangular_256x1024_seed42'
FINAL=OUT/'checkpoints/final.ckpt'
LAST=OUT/'checkpoints/last.ckpt'
CALIBRATION=ROOT/'experiments/phase13a_rectangular_256x1024/calibration_report_r1_retry1.json'
B0_CONFIG=ROOT/'experiments/phase07b_baseline256_seed42/config.json'
MIB=1024**2


def now():return datetime.now().astimezone().isoformat()


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_json(path,data):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def protocol():
    return dict(experiment='Rectangular_EfficientAD_256x1024',dataset='MVTec AD 2',category='sheet_metal',
        dataset_root=str(ROOT/'datasets/MVTec_AD_2'),native_size_hw=[1056,4224],input_size_hw=list(cal.TARGET),
        aspect_ratio=4.,model='EfficientAD-S',model_size='small',steps=70000,max_steps=70000,max_epochs=1000,
        batch=1,batch_size=1,num_workers=0,seed=42,precision='FP32',trainer_precision='32-true',
        padding=False,pad_maps=True,total_parameters=8058628,optimizer='Adam',learning_rate=1e-4,weight_decay=1e-5,
        scheduler='StepLR',scheduler_step_size=66500,scheduler_gamma=0.1,early_stopping=False,
        validation_during_fit=False,final_full_validation=True,evaluator=False,visualizer=False,post_processor=False,
        difference_from_B0='aspect-ratio-preserving rectangular input only',no_tile=True,no_halo=True,
        pre_processor='EfficientAd.configure_pre_processor(image_size=(256,1024))',
        predicted_training_hours_steady=10.8939,predicted_training_hours_total=11.7258)


def trainer_config():
    return dict(max_steps=70000,max_epochs=1000,accelerator='gpu',devices=1,precision='32-true',
        num_sanity_val_steps=0,limit_train_batches=1.0,limit_val_batches=0,enable_checkpointing=True,
        log_every_n_steps=1,enable_progress_bar=False)


def initial_report():
    return dict(status='NOT_RUN',timestamp_start=None,timestamp_end=None,target_steps=70000,completed_steps=0,
        global_step=0,final_epoch=None,training_elapsed_seconds=None,training_elapsed_minutes=None,
        training_elapsed_hours=None,seconds_per_step=None,session_training_elapsed_seconds=None,
        peak_gpu_allocated_mib=None,peak_gpu_reserved_mib=None,loss_nan_detected=False,loss_inf_detected=False,
        oom_detected=False,final_validation_batches=0,final_validation_completed=False,final_quantiles=None,
        final_quantiles_finite=False,checkpoint_path=None,checkpoint_sha256=None,strict_reload='NOT_RUN',
        resumed=False,resume_from_step=0,log_paths=[],warnings=[],last_verified_step=0,
        timing_note='Fit setup and periodic checkpoint validation included; final validation and reload excluded. Resume sums saved elapsed time and current session; memory peaks cover this process.')


def config_signature(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True,ensure_ascii=True,allow_nan=False).encode('utf-8')).hexdigest()


def session_log_paths(prior,current,resume_from_step):
    # At max_steps Lightning skips fit and does not create a new metrics.csv.
    return list(prior) if resume_from_step==70000 else list(prior)+[current]


def validate_resume(saved,config):
    meta=saved.get('phase13b',{})
    if meta.get('protocol')!=protocol() or meta.get('config_signature')!=config_signature(config):
        raise ValueError('FAIL_RESUME_PROTOCOL')
    step=saved.get('global_step')
    if type(step) is not int or not 0<step<=70000:raise ValueError('FAIL_RESUME_STEP')
    if any(not saved.get(k) for k in ('state_dict','optimizer_states','lr_schedulers','callbacks')):
        raise ValueError('FAIL_RESUME_INCOMPLETE')
    elapsed=meta.get('training_elapsed_seconds')
    if not isinstance(elapsed,(int,float)) or not math.isfinite(elapsed) or elapsed<0:raise ValueError('FAIL_RESUME_TIMING')
    if not meta.get('timestamp_start') or not meta.get('log_paths'):raise ValueError('FAIL_RESUME_PROVENANCE')
    state=saved['callbacks'].get('BaselineMonitor')
    if state is None or state.get('nan_detected') or state.get('inf_detected'):raise ValueError('FAIL_RESUME_NUMERICS')
    return step,meta


def validate_shapes(g):
    expected=dict(teacher_feature_shape=[1,384,56,248],student_feature_shape=[1,768,56,248],
        student_st_shape=[1,384,56,248],st_discrepancy_shape=[1,384,56,248],ae_feature_shape=[1,384,56,248],
        student_ae_half_shape=[1,384,56,248],final_anomaly_map_shape=[1,1,256,1024])
    if any(g.get(k)!=v for k,v in expected.items()):raise ValueError('FAIL_SHAPE_GATE')
    if g.get('encoder_layers',{}).get('enconv6',{}).get('output')!=[1,64,1,25]:raise ValueError('FAIL_SHAPE_GATE: latent')
    if not g.get('st_shape_match') or not g.get('ae_shape_match'):raise ValueError('FAIL_SHAPE_GATE: branch matching')


def final_checkpoint_gate(step,report,quantiles):
    if step!=70000:raise ValueError('FAIL_FINAL_STEP')
    if not report.get('final_validation_completed') or report.get('final_validation_batches')!=19:
        raise ValueError('FAIL_FINAL_VALIDATION')
    if set(quantiles)!={'qa_st','qb_st','qa_ae','qb_ae'} or not all(math.isfinite(v) for v in quantiles.values()):
        raise ValueError('FAIL_QUANTILE_STATE')
    if quantiles['qb_st']<=quantiles['qa_st'] or quantiles['qb_ae']<=quantiles['qa_ae']:
        raise ValueError('FAIL_QUANTILE_STATE: nonpositive range')


class BaselineMonitor(VerifiedMonitor):
    """Reuse Phase08B loss/state/pickle mechanism; Phase13B provenance and gates."""
    def on_save_checkpoint(self,trainer,pl_module,checkpoint):
        super().on_save_checkpoint(trainer,pl_module,checkpoint)
        meta=checkpoint.pop('phase08b');meta['protocol']=protocol()
        meta['config_signature']=config_signature(self.config);checkpoint['phase13b']=meta

    def on_train_start(self,trainer,pl_module):
        if trainer.max_steps!=70000 or trainer.precision!='32-true' or trainer.limit_val_batches!=0:
            raise ValueError('FAIL_TRAINING_PROTOCOL')
        if any(isinstance(c,EarlyStopping) for c in trainer.callbacks):raise ValueError('FAIL_EARLY_STOPPING')

    def on_train_batch_start(self,trainer,pl_module,batch,batch_idx):
        cal.shape_gate(batch.image)
        if int(trainer.global_step)>=70000:raise ValueError('FAIL_STEP_OVERRUN')
        step=int(trainer.global_step)
        if step and step%1000==0 and self.report['last_verified_step']<step:
            saved=torch.load(LAST,map_location='cpu',weights_only=False)
            stored,_=validate_resume(saved,self.config);del saved
            if stored!=step:raise ValueError('FAIL_LAST_CHECKPOINT_STEP')
            self.report['last_verified_step']=step
            write_json(OUT/'last_checkpoint_audit.json',dict(status='PASS',global_step=step,checked=now(),size_bytes=LAST.stat().st_size))

    def on_before_backward(self,trainer,pl_module,loss):
        super().on_before_backward(trainer,pl_module,loss)
        values={}
        for key in ('train_st','train_ae','train_stae','train_loss'):
            v=trainer.callback_metrics.get(key+'_step',trainer.callback_metrics.get(key))
            if v is None:raise ValueError('FAIL_LOSS_LOGGING: '+key)
            value=float(v.detach().cpu()) if torch.is_tensor(v) else float(v)
            self.report['loss_nan_detected']|=math.isnan(value);self.report['loss_inf_detected']|=math.isinf(value)
            if not math.isfinite(value):raise ValueError('FAIL_NUMERICS: '+key)
            values[key]=value
        self.report['last_losses']=values

    def on_train_batch_end(self,trainer,pl_module,outputs,batch,batch_idx):
        super().on_train_batch_end(trainer,pl_module,outputs,batch,batch_idx)
        step=int(trainer.global_step)
        if step==1 or step%100==0:
            elapsed=self.timing['prior']+time.perf_counter()-self.timing['start']
            self.report.update(status='TRAINING',global_step=step,completed_steps=step,final_epoch=int(trainer.current_epoch),
                progress_timestamp=now(),training_elapsed_seconds=elapsed,training_elapsed_hours=elapsed/3600,
                seconds_per_step=elapsed/step,estimated_remaining_hours=(70000-step)*elapsed/step/3600,
                peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/MIB,peak_gpu_reserved_mib=torch.cuda.max_memory_reserved()/MIB)
            write_json(OUT/'training_report.json',self.report)
            print(f'Phase13B {step}/70000; elapsed={elapsed/3600:.4f}h; loss={self.last_loss:.6f}; timestamp={now()}',flush=True)

    def on_validation_start(self,trainer,pl_module):
        if trainer.global_step!=70000:raise ValueError('FAIL_VALIDATION_BEFORE_TARGET')

    def on_test_start(self,trainer,pl_module):raise ValueError('FAIL_TEST_FORBIDDEN')


def environment_snapshot(versions):
    lines=[f'Snapshot: {now()}',f'Python executable: {sys.executable}']+[f'{k}: {v}' for k,v in versions.items()]
    for command in ([sys.executable,'-m','pip','freeze'],['nvidia-smi','--query-gpu=index,name,driver_version,memory.total','--format=csv']):
        r=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=45,
            check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        lines+=['\n$ '+' '.join(command),r.stdout,r.stderr]
    (OUT/'environment.txt').write_text('\n'.join(lines),encoding='utf-8')


def audit_data(dm,data):
    for split,dataset in [('train',dm.train_data),('validation',dm.val_data)]:
        base=(ROOT/'datasets/MVTec_AD_2/sheet_metal'/split/'good').resolve()
        if not (dataset.samples.label_index==0).all():raise ValueError('FAIL_DATA_LABEL: '+split)
        if any(not Path(p).resolve().is_relative_to(base) for p in dataset.samples.image_path):raise ValueError('FAIL_DATA_SPLIT: '+split)
    records=[]
    for split,dataset in [('train',dm.train_data),('validation',dm.val_data),('test_public',dm.test_public_data)]:
        records += [(split,str(Path(p).resolve()),int(label)) for p,label in zip(dataset.samples.image_path,dataset.samples.label_index)]
    return hashlib.sha256(json.dumps(sorted(records)).encode()).hexdigest()


def verify_b0(config,model):
    b0=json.loads(B0_CONFIG.read_text(encoding='utf-8'))
    keys=('model_size','batch_size','num_workers','seed','max_steps','max_epochs','optimizer','learning_rate','weight_decay',
        'scheduler','scheduler_step_size','scheduler_gamma','validation_during_fit','final_full_validation','early_stopping','evaluator','visualizer','post_processor')
    if any(config[k]!=b0[k] for k in keys):raise ValueError('FAIL_B0_TRAINING_PROTOCOL')
    counts=dict(parameter_count_total=sum(p.numel() for p in model.parameters()),
        parameter_count_requires_grad=sum(p.numel() for p in model.parameters() if p.requires_grad),
        optimizer_parameter_count=sum(p.numel() for p in model.model.student.parameters())+sum(p.numel() for p in model.model.ae.parameters()))
    if counts['parameter_count_total']!=8058628 or any(v!=b0[k] for k,v in counts.items()):raise ValueError('FAIL_PARAMETERS')
    if model.lr!=b0['learning_rate'] or model.weight_decay!=b0['weight_decay']:raise ValueError('FAIL_OPTIMIZER')
    return counts


def summary(r):
    g=r.get('feature_shapes',{})
    lines=['='*40,'PHASE 13B - RECTANGULAR FORMAL TRAINING','='*40,
        'Dataset: MVTec AD 2 / sheet_metal','Method: Aspect-Ratio-Preserving EfficientAD-S',
        'Native: 1056x4224','Input: 256x1024','Aspect ratio: 4.000','Model: EfficientAD-S','Seed: 42','Batch: 1','FP32',
        '-'*40,'FEATURE SHAPES']
    for key in ('teacher_feature_shape','student_feature_shape','student_st_shape','ae_feature_shape'):
        lines.append(f'{key}: {g.get(key,"NOT_RUN")}')
    lines.append('AE latent: '+str(g.get('encoder_layers',{}).get('enconv6',{}).get('output','NOT_RUN')))
    lines += ['Shape gate: '+str(r.get('shape_gate','NOT_RUN')),'-'*40,'TRAINING','Target steps: 70000']
    for key in ('completed_steps','training_elapsed_hours','seconds_per_step','peak_gpu_allocated_mib','peak_gpu_reserved_mib',
        'loss_nan_detected','loss_inf_detected','oom_detected','actual_vs_projected_ratio_steady','actual_vs_projected_ratio_total'):
        lines.append(f'{key}: {r.get(key)}')
    lines += ['-'*40,'VALIDATION','Validation images: '+str(r.get('final_validation_batches')),
        'Final quantiles finite: '+str(r.get('final_quantiles_finite')),'-'*40,'CHECKPOINT']
    for key in ('checkpoint_path','checkpoint_sha256','strict_reload','resumed','resume_from_step','total_parameters'):
        lines.append(f'{key}: {r.get(key)}')
    lines += ['STATUS: '+r['status'],str(r.get('error_message','')),'='*40]
    return '\n'.join(lines)


def finish_validation_and_save(engine,model,dm,r,config):
    if engine.trainer.global_step!=70000:raise ValueError('FAIL_FINAL_STEP')
    engine.trainer.limit_val_batches=1.0;r['final_validation_batches']=0
    start=time.perf_counter()
    engine.trainer.validate(model=model,datamodule=dm,ckpt_path=None,verbose=False)
    torch.cuda.synchronize();r['validation_elapsed_seconds']=time.perf_counter()-start
    q={k:float(v.detach().cpu()) for k,v in model.model.quantiles.items()}
    r['final_validation_completed']=True
    final_checkpoint_gate(int(engine.trainer.global_step),r,q)
    r.update(final_quantiles=q,final_quantiles_finite=True)
    if FINAL.exists():raise FileExistsError('FAIL_FINAL_EXISTS')
    engine.trainer.save_checkpoint(str(FINAL),weights_only=False)
    saved=torch.load(FINAL,map_location='cpu',weights_only=False);step,meta=validate_resume(saved,config)
    if step!=70000 or not meta['final_validation_completed']:raise ValueError('FAIL_FINAL_PROVENANCE')
    if any(float(saved['state_dict']['model.quantiles.'+k])!=v for k,v in q.items()):raise ValueError('FAIL_SAVED_QUANTILES')
    del saved
    expected=cal.parameter_hash(model);model.cpu();gc.collect();torch.cuda.empty_cache()
    restored=EfficientAd.load_from_checkpoint(str(FINAL),map_location='cpu',strict=True,weights_only=False)
    if cal.parameter_hash(restored)!=expected:raise ValueError('FAIL_STRICT_RELOAD_PARAMETERS')
    if list(restored.pre_processor.transform.transforms[0].size)!=[256,1024]:raise ValueError('FAIL_RELOADED_PREPROCESSOR')
    restored.eval().to('cuda:0')
    with torch.no_grad():x=restored.pre_processor(read_image(dm.train_data.samples.image_path.iloc[0],as_tensor=True)[None].to('cuda:0'))
    g=cal.probe(restored,x);validate_shapes(g)
    if any(float(restored.model.quantiles[k].detach().cpu())!=v for k,v in q.items()):raise ValueError('FAIL_RELOADED_QUANTILES')
    r.update(strict_reload='PASS',reloaded_feature_shapes=g,checkpoint_path=str(FINAL),checkpoint_sha256=digest(FINAL),
        checkpoint_size_bytes=FINAL.stat().st_size)
    del x,restored;gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--precheck-only',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    os.chdir(ROOT);OUT.mkdir(exist_ok=True,parents=True);(OUT/'checkpoints').mkdir(exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
    if FINAL.exists() or (LAST.exists() and not args.resume):
        print('Refusing overwrite/restart: final exists or last requires --resume',flush=True);return 1
    if not args.precheck_only and not args.resume and (OUT/'training_report.json').exists():
        print('Existing training report: refusing restart; inspect or resume last.ckpt',flush=True);return 1
    r=initial_report();r['timestamp_start']=now();config=protocol();timing=dict(prior=0.,start=None,finished=None)
    engine=logger=model=None;stage='PRECHECK';code=1;memory=False
    try:
        torch.set_num_threads(4);cal.seed()
        if not torch.cuda.is_available():raise ValueError('FAIL_GPU')
        torch.cuda.set_device(0)
        versions=dict(python=platform.python_version(),torch=str(torch.__version__),torchvision=torchvision.__version__,
            anomalib=version('anomalib'),lightning=version('lightning'),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0))
        if versions['anomalib']!='2.6.1':raise ValueError('FAIL_ANOMALIB_VERSION')
        config['framework_versions']=versions;r['gpu_total_memory_mib']=torch.cuda.get_device_properties(0).total_memory/MIB
        prior=json.loads(CALIBRATION.read_text(encoding='utf-8'))
        if prior.get('status')!='PASS' or not prior.get('formal_training_feasible') or prior['calibration']['completed_steps']!=500:
            raise ValueError('FAIL_CALIBRATION_REFERENCE')
        config.update(calibration_report=str(CALIBRATION),calibration_sha256=digest(CALIBRATION))
        dm,data=cal.dataset();config['data_fingerprint']=audit_data(dm,data);r['data']=data
        print('Data gate PASS: train137 validation19 test_public114 normal24 anomaly90; public metadata only',flush=True)
        model,pre=cal.make_model();counts=verify_b0(config,model);config.update(counts)
        config['preprocessor_configuration']=repr(model.pre_processor.transform)
        r.update(total_parameters=counts['parameter_count_total'],trainable_parameters=counts['parameter_count_requires_grad'])
        if not math.isclose(cal.scaling()['scale_x'],cal.scaling()['scale_y']):raise ValueError('FAIL_DIMENSION_ORDER')
        print('Input H/W PASS: Resize H=256 W=1024; anisotropy=1',flush=True)
        model.prepare_pretrained_model();model.eval().to('cuda:0');r['input_checks']=cal.input_checks(model,data['train_paths'])
        with torch.no_grad():x=model.pre_processor(read_image(data['train_paths'][0],as_tensor=True)[None].to('cuda:0'))
        g=cal.probe(model,x);validate_shapes(g);r.update(feature_shapes=g,shape_gate='PASS');del x
        r['imagenette']=cal.imagenette_probe(model)
        print('Shape gate PASS; Teacher384x56x248 Student768x56x248 ST384x56x248 AE latent64x1x25 AE384x56x248',flush=True)
        print('ImageNette rectangular PASS; parameters8058628 PASS; GPU PASS',flush=True)
        write_json(OUT/'feature_shapes.json',g);write_json(OUT/'preprocessor_audit.json',pre)
        environment_snapshot(versions)
        del model;model=None;gc.collect();torch.cuda.empty_cache();cal.seed();model,_=cal.make_model();dm,_=cal.dataset()
        if args.resume:
            saved=torch.load(LAST,map_location='cpu',weights_only=False);step,meta=validate_resume(saved,config)
            r.update(resumed=True,resume_from_step=step,timestamp_start=meta['timestamp_start'],log_paths=meta['log_paths'],last_verified_step=step)
            timing['prior']=meta['training_elapsed_seconds'];del saved
        monitor=BaselineMonitor(r,config,timing)
        restored=pickle.loads(pickle.dumps(monitor))
        if restored.state_key!='BaselineMonitor' or type(restored) is not BaselineMonitor:raise ValueError('FAIL_CALLBACK_PICKLE')
        del restored
        recovery=ModelCheckpoint(dirpath=str(OUT/'checkpoints'),monitor=None,save_top_k=0,save_last=True,
            every_n_train_steps=1000,save_on_train_epoch_end=False,enable_version_counter=False)
        pickle.loads(pickle.dumps(recovery))
        tree=ast.parse(Path(__file__).read_text(encoding='utf-8'))
        if any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in {'test','test_dataloader','predict'} for n in ast.walk(tree)):
            raise ValueError('FAIL_TEST_PATH')
        print('Checkpoint callback PASS: official ModelCheckpoint + inherited module-level BaselineMonitor; no test path PASS',flush=True)
        write_json(OUT/'config.json',config);r['precheck']='PASS'
        if args.precheck_only:r['status']='PRECHECK_PASS';code=0;return code
        logger=CSVLogger(save_dir=str(OUT/'logs'),name='baseline',flush_logs_every_n_steps=100)
        r['log_paths']=session_log_paths(r['log_paths'],str(Path(logger.log_dir)/'metrics.csv'),r['resume_from_step'])
        logger.log_hyperparams(config)
        engine=Engine(**trainer_config(),default_root_dir=str(OUT),logger=logger,
            callbacks=[monitor,LearningRateMonitor(logging_interval='step'),recovery])
        stage='TRAINING';torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();memory=True;torch.cuda.synchronize()
        timing['start']=time.perf_counter();r['status']='TRAINING';write_json(OUT/'training_report.json',r)
        engine.fit(model=model,datamodule=dm,ckpt_path=str(LAST) if args.resume else None)
        torch.cuda.synchronize();timing['finished']=timing['prior']+time.perf_counter()-timing['start']
        r.update(completed_steps=int(engine.trainer.global_step),global_step=int(engine.trainer.global_step),final_epoch=int(engine.trainer.current_epoch),
            peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/MIB,peak_gpu_reserved_mib=torch.cuda.max_memory_reserved()/MIB)
        if r['completed_steps']!=70000:raise ValueError('FAIL_STEP_COUNT')
        stage='LOGGING';logger.save();check_logs(r['log_paths'],r)
        stage='LAST_CHECKPOINT';saved=torch.load(LAST,map_location='cpu',weights_only=False);step,_=validate_resume(saved,config);del saved
        if step!=70000:raise ValueError('FAIL_LAST_CHECKPOINT_STEP')
        r['last_verified_step']=step;r['last_checkpoint_loadable']=True
        stage='FINAL_VALIDATION_CHECKPOINT';finish_validation_and_save(engine,model,dm,r,config)
        if r['loss_nan_detected'] or r['loss_inf_detected'] or r['oom_detected']:raise ValueError('FAIL_NUMERICS')
        r['status']='PASS';code=0
    except (Exception,KeyboardInterrupt) as exc:
        r.update(status='FAIL',failure_stage=stage,error_message=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),
            oom_detected=isinstance(exc,torch.cuda.OutOfMemoryError) or 'cuda out of memory' in str(exc).lower())
        traceback.print_exc()
    finally:
        if timing['start'] is not None:
            elapsed=timing['finished'] if timing['finished'] is not None else timing['prior']+time.perf_counter()-timing['start']
            r.update(training_elapsed_seconds=elapsed,training_elapsed_minutes=elapsed/60,training_elapsed_hours=elapsed/3600,
                actual_vs_projected_ratio_steady=elapsed/3600/protocol()['predicted_training_hours_steady'],
                actual_vs_projected_ratio_total=elapsed/3600/protocol()['predicted_training_hours_total'])
            if r['completed_steps']:r['seconds_per_step']=elapsed/r['completed_steps']
        if memory and r['peak_gpu_allocated_mib'] is None:
            r.update(peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/MIB,peak_gpu_reserved_mib=torch.cuda.max_memory_reserved()/MIB)
        if r['peak_gpu_allocated_mib'] is not None:r['fraction_allocated']=r['peak_gpu_allocated_mib']/r['gpu_total_memory_mib']
        if logger is not None:logger.save()
        r['timestamp_end']=now();stem='precheck_report' if args.precheck_only else 'training_report'
        write_json(OUT/(stem+'.json'),r);(OUT/(stem+'.txt')).write_text(summary(r),encoding='utf-8')
        print(summary(r),flush=True)
    return code


if __name__=='__main__':
    # Stable module identity is required by Lightning checkpoint serialization.
    from scripts.train_rectangular_efficientad_256x1024 import main as run_main
    raise SystemExit(run_main())
