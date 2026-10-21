"""Train-normal-only initial scale calibration and bounded P2-S1 smoke runs."""
import ast
import csv
import gc
import inspect
import json
import math
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from anomalib.models import EfficientAd
from anomalib.engine import Engine
from lightning.pytorch import Callback
from src.models import multiscale_st_s1 as s1
from src.evaluation import rectangular_efficientad_calibration as base
from src.evaluation.efficientad_branch_audit import parameter_hash, state_hash
from src.evaluation.rectangular_formal_evaluation import input_tensor as normal_input

ROOT = base.ROOT
OUT = ROOT / 'experiments/phase14b_a_s1_calibration'
seed = base.seed
LOG_FIELDS = ['phase', 'step', 'epoch', 'learning_rate', 'official_total', 'final_st',
              'official_st_with_penalty', 'ae', 'stae', 'penalty', 's1_raw', 'weighted_s1',
              'p2_total', 'aux_fraction', 'lambda_s1', 'student_forward_calls']


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def normal_paths(split):
    if split not in ('train', 'validation'):
        raise ValueError('FAIL_NORMAL_SOURCE')
    # Reuse the validated official split membership, rather than infer paths.
    dm, _ = base.dataset()
    data = dm.train_data if split == 'train' else dm.val_data
    if not (data.samples.label_index == 0).all():
        raise ValueError('FAIL_NORMAL_SOURCE')
    paths = sorted(Path(p) for p in data.samples.image_path.tolist())
    if len(paths) != (137 if split == 'train' else 19):
        raise ValueError('FAIL_NORMAL_SOURCE_COUNT')
    return paths


def fresh_model():
    seed()
    # Reuse preprocessor and cached pretrained asset gates without maintaining
    # another rectangular transform. Reset the seed before fresh module creation.
    reference, metadata = base.make_model()
    pre = reference.pre_processor
    del reference
    seed()
    model = s1.MultiScaleSTS1(model_size='small', imagenet_dir='./datasets/imagenette',
                            pre_processor=pre, evaluator=False, post_processor=False, visualizer=False)
    model.prepare_pretrained_model()
    if sum(p.numel() for p in model.parameters()) != 8058628:
        raise ValueError('FAIL_PARAMETERS')
    if not torch.cuda.is_available():
        raise ValueError('FAIL_GPU')
    model.eval().to('cuda:0')
    metadata.update(initialization='seed42 fresh Student and AE; official pretrained Teacher only',
                    baseline_checkpoint_loaded=False, new_trainable_parameters=0,
                    student_initial_hash=parameter_hash(model.model.student),
                    ae_initial_hash=parameter_hash(model.model.ae),
                    teacher_initial_hash=parameter_hash(model.model.teacher))
    return model, metadata


def implementation_audit():
    from importlib.metadata import version
    from anomalib.models.image.efficient_ad.torch_model import EfficientAdModel
    targets = {name: getattr(EfficientAd, name) for name in
               ('training_step', 'on_train_start', 'teacher_channel_mean_std',
                'prepare_imagenette_data', 'configure_optimizers',
                '_get_quantiles_of_maps', 'map_norm_quantiles')}
    targets.update({name: getattr(EfficientAdModel, name) for name in
                    ('compute_student_teacher_distance', 'compute_losses', 'compute_maps', 'forward')})
    targets['reduce_tensor_elems'] = s1.reduce_tensor_elems
    sources = {k: dict(signature=str(inspect.signature(v)), source=inspect.getsource(v),
                       file=inspect.getsourcefile(v), line=inspect.getsourcelines(v)[1]) for k, v in targets.items()}
    loss_source = sources['compute_losses']['source']
    if version('anomalib') != '2.6.1' or '0.999' not in loss_source:
        raise ValueError('FAIL_OFFICIAL_LOSS_API')
    if '0.995' not in sources['_get_quantiles_of_maps']['source']:
        raise ValueError('FAIL_OFFICIAL_QUANTILE_API')
    return dict(anomalib=version('anomalib'), functions=sources,
                official_st_loss_formula='hard((normalized_teacher - student_first384)^2) + mean(student(ImageNette)_first384^2)',
                hard_mining_rule='Direct official reduce_tensor_elems, q=0.999, mean of values >= quantile',
                penalty_rule='Exactly one original final-ST ImageNette penalty; none added to S1',
                final_total='official ST_with_penalty + AE + STAE',
                p2_total='unchanged official total + frozen lambda_s1 * S1 hard loss',
                s1_teacher_normalization='Raw post-ReLU feature; detached; final teacher mean/std remains official final-only',
                s1_hook='Phase14A Capture: avgpool1 forward_pre_hook input, conv1.post_relu',
                s1_discrepancy_elements=128*253*1021, quantile_max_elements=2**24,
                s1_sampling='Official reduce_tensor_elems random subsampling, seed42; no substitute approximation rule',
                validation_qa_probability=0.9, validation_qb_probability=0.995,
                normalization='0.1 * (map - qa) / (qb - qa)',
                official_combined='0.5 * final_ST_norm + 0.5 * AE_norm',
                multiscale_ST='0.5 * resized_S1_norm + 0.5 * final_ST_norm',
                candidate_combined='0.5 * multiscale_ST + 0.5 * official_AE_norm',
                inference_roles=['NEW_MODEL_FINAL_ONLY', 'NEW_MODEL_MS_ST_ONLY', 'NEW_MODEL_MS_COMBINED'],
                test_data_used=False, site_packages_modified=False)


def protocol_gate():
    # Inspect actual project execution paths, not the narrative comments.
    sources = [inspect.getsource(s1), inspect.getsource(fresh_model),
               inspect.getsource(calibrate_lambda), inspect.getsource(train_run)]
    forbidden = {'test', 'test_step', 'predict', 'auroc', 'aupro', 'f1', 'load_from_checkpoint'}
    for source in sources:
        tree = ast.parse(textwrap.dedent(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, 'id', '')
                if name.lower() in forbidden:
                    return False
    return (s1.MultiScaleSTS1.configure_optimizers is EfficientAd.configure_optimizers
            and s1.MultiScaleSTS1.map_norm_quantiles is EfficientAd.map_norm_quantiles)


def dry_run_has_no_update():
    tree = ast.parse(textwrap.dedent(inspect.getsource(calibrate_lambda)))
    return not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr in ('backward', 'step', 'fit', 'configure_optimizers') for n in ast.walk(tree))


def trainer_options(steps):
    return base.trainer_config(steps)


def fresh_training_model(value):
    model, initialization = fresh_model()
    model.freeze_lambda(value)
    # Lightning does not undo a caller's explicit eval() at initial fit entry.
    model.train()
    return model, initialization


def precheck():
    _, data = base.dataset()
    audit = implementation_audit()
    model, initialization = fresh_model()
    x = normal_input(model, normal_paths('train')[0])
    before, state_before = parameter_hash(model), state_hash(model)
    with torch.no_grad():
        t, s, student, d = s1.s1_features(model, x)
        final, ae = model.model.compute_maps(x, student, d, normalize=False)
        official = model.model(x).anomaly_map
    if not torch.equal(0.5*final+0.5*ae, official):
        raise ValueError('FAIL_OFFICIAL_MAP_RECONSTRUCTION')
    if not protocol_gate() or not dry_run_has_no_update():
        raise ValueError('FAIL_PROTOCOL')
    if before != parameter_hash(model) or state_before != state_hash(model):
        raise ValueError('FAIL_STATE_MUTATION')
    model.prepare_imagenette_data(s1.TARGET)
    extra = next(iter(model.imagenet_loader))[0]
    s1.input_gate(extra)
    result = dict(status='PASS', data=data, input_hw=list(s1.TARGET), s1_teacher_shape=list(t.shape),
                  s1_student_shape=list(s.shape), final_st_shape=list(d.shape),
                  student_shape=list(student.shape), output_shape=list(official.shape),
                  parameters=8058628, no_projector=True, no_test_metric=True,
                  official_loss_audit=True, imagenette_shape=list(extra.shape),
                  parameter_state_unchanged=True, initialization=initialization)
    save('implementation_audit.json', audit)
    save('precheck_report.json', result)
    print('PRECHECK PASS: input=(256,1024); S1=128x253x1021; Final=384x56x248; parameters=8058628; no projector; no test metric; official loss audit PASS', flush=True)
    del model, x, t, s, student, d, final, ae, official, extra
    gc.collect(); torch.cuda.empty_cache()
    return result


@torch.no_grad()
def calibrate_lambda():
    model, initialization = fresh_model()
    paths = normal_paths('train')
    # Mirror on_train_start order and fit the official final Teacher statistics
    # on all 137 normal samples, with the official rectangular preprocessing.
    model.prepare_imagenette_data(s1.TARGET)
    class NormalBatches:
        def __len__(self): return len(paths)
        def __iter__(self):
            for path in paths:
                yield SimpleNamespace(image=normal_input(model, path))
    model.model.mean_std.update(model.teacher_channel_mean_std(NormalBatches()))
    before, state_before = parameter_hash(model), state_hash(model)
    rows = []
    for i, path in enumerate(paths[:32]):
        x = normal_input(model, path)
        t, s, _, distance = s1.s1_features(model, x)
        final_loss = float(s1.hard_loss(distance))
        shallow_loss = float(s1.hard_loss(s1.discrepancy(t, s)))
        rows.append(dict(index=i, sample_id=path.stem, path=str(path), label=0,
                         final_st_distillation=final_loss, s1_raw=shallow_loss))
        print(f'LOSS SCALE {i+1}/32: final={final_loss:.8f}, S1={shallow_loss:.8f}', flush=True)
        del x, t, s, distance
    result = s1.loss_ratio([r['final_st_distillation'] for r in rows], [r['s1_raw'] for r in rows],
                          split='train', labels=[0]*32)
    result.update(rows=rows, initialization=initialization, teacher_statistics_samples=137,
                  teacher_statistics_protocol='Official teacher_channel_mean_std over rectangular normal train inputs',
                  teacher_mean=model.model.mean_std['mean'].flatten().cpu().tolist(),
                  teacher_std=model.model.mean_std['std'].flatten().cpu().tolist(),
                  parameters_before=before, parameters_after=parameter_hash(model),
                  state_before=state_before, state_after=state_hash(model), optimizer_steps=0,
                  sample_order='Ascending official normal train image path, first32',
                  lambda_frozen=result['safety_gate']=='PASS')
    if before != result['parameters_after'] or state_before != result['state_after']:
        raise ValueError('FAIL_DRY_RUN_MUTATION')
    save('loss_scale_calibration.json', result)
    del model
    gc.collect(); torch.cuda.empty_cache()
    return result


class CalibrationMonitor(Callback):
    def __init__(self, phase, target, value):
        self.phase, self.target, self.value = phase, target, value
        self.rows = []; self.warm_end = None; self.end = None

    def on_train_start(self, trainer, pl_module):
        if trainer.max_steps != self.target or trainer.precision != '32-true':
            raise ValueError('FAIL_TRAINING_PROTOCOL')
        if trainer.limit_val_batches != 0 or trainer.checkpoint_callbacks:
            raise ValueError('FAIL_TRAINING_PROTOCOL')

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if any(p.grad is not None for p in pl_module.model.teacher.parameters()):
            raise ValueError('FAIL_TEACHER_GRAD')
        for p in pl_module.parameters():
            if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
                raise ValueError('FAIL_NUMERICS: gradient')

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if float(pl_module.lambda_s1) != self.value or step != len(self.rows)+1 or step > self.target:
            raise ValueError('FAIL_TRAINING_STATE')
        s1.input_gate(batch.image)
        row = dict(phase=self.phase, step=step, epoch=int(trainer.current_epoch),
                   learning_rate=trainer.optimizers[0].param_groups[0]['lr'], **pl_module.last_loss_log)
        if not np.isfinite([v for k,v in row.items() if k != 'phase']).all():
            raise ValueError('FAIL_NUMERICS')
        self.rows.append(row)
        with (OUT/'training_log.csv').open('a', encoding='utf-8', newline='') as stream:
            csv.DictWriter(stream, fieldnames=LOG_FIELDS).writerow(row)
        if step in (50, self.target):
            torch.cuda.synchronize()
            now = time.perf_counter()
            if step == 50: self.warm_end = now
            if step == self.target: self.end = now
        if step <= 3 or step % 25 == 0:
            print(f'{self.phase} {step}/{self.target}: total={row["p2_total"]:.6f}, final={row["final_st"]:.6f}, S1={row["s1_raw"]:.6f}, weighted={row["weighted_s1"]:.6f}', flush=True)


def train_run(steps, value):
    options = trainer_options(steps)
    # Every run has a separately seeded, fresh official initialization.
    model, initialization = fresh_training_model(value)
    dm, _ = base.dataset()
    branches = ('teacher', 'student', 'ae')
    before = {k: parameter_hash(getattr(model.model,k)) for k in branches}
    conv1 = parameter_hash(model.model.student.conv1)
    quantiles = {k:v.detach().cpu().clone() for k,v in model.model.quantiles.items()}
    monitor = CalibrationMonitor(f'steps{steps}', steps, value)
    report = dict(status='RUNNING', target_steps=steps, completed_steps=0, NaN=False, Inf=False, OOM=False,
                  initialization=initialization, lambda_s1=value, validation_batches=0, test_forwards=0)
    engine = Engine(**options, default_root_dir=str(OUT), callbacks=[monitor])
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        engine.fit(model=model, datamodule=dm)
        torch.cuda.synchronize()
        report['total_fit_seconds'] = time.perf_counter()-start
        report['completed_steps'] = int(engine.trainer.global_step)
        if report['completed_steps'] != steps or len(monitor.rows) != steps:
            raise ValueError('FAIL_STEP_COUNT')
        after = {k: parameter_hash(getattr(model.model,k)) for k in branches}
        report.update(branch_hashes_before=before, branch_hashes_after=after,
                      teacher_unchanged=before['teacher']==after['teacher'],
                      student_updated=before['student']!=after['student'], ae_updated=before['ae']!=after['ae'],
                      student_conv1_updated=conv1!=parameter_hash(model.model.student.conv1),
                      lambda_unchanged=float(model.lambda_s1)==value,
                      parameters=sum(p.numel() for p in model.parameters()),
                      losses_finite=True, last_losses=monitor.rows[-1])
        if not all(report[k] for k in ('teacher_unchanged','student_updated','ae_updated','student_conv1_updated','lambda_unchanged')):
            raise ValueError('FAIL_STATE_AUDIT')
        if report['parameters'] != 8058628: raise ValueError('FAIL_PARAMETERS')
        if any(not torch.equal(v, model.model.quantiles[k].cpu()) for k,v in quantiles.items()) or bool(model.s1_quantiles_fitted):
            raise ValueError('FAIL_UNEXPECTED_QUANTILE_UPDATE')
        if steps == 500:
            steady = (monitor.end-monitor.warm_end)/450
            report.update(steady_seconds_per_step=steady, steps_per_second=1/steady,
                          projected_70000_hours=steady*70000/3600,
                          projected_70000_hours_total_average=report['total_fit_seconds']/500*70000/3600)
            fields = ('final_st','s1_raw','weighted_s1','p2_total')
            report['first50_medians'] = {k:float(np.median([r[k] for r in monitor.rows[:50]])) for k in fields}
            report['last50_medians'] = {k:float(np.median([r[k] for r in monitor.rows[-50:]])) for k in fields}
        report['status'] = 'PASS'
    except Exception as exc:
        report.update(status='FAIL', error=str(exc), completed_steps=len(monitor.rows),
                      OOM=isinstance(exc,torch.cuda.OutOfMemoryError))
        raise
    finally:
        report.update(peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                      peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
        if report['peak_allocated_mib'] > 6500:
            report.update(status='FAIL', error='FAIL_MEMORY_LIMIT')
        save(f'training_{steps}_report.json', report)
        del engine, model
        gc.collect(); torch.cuda.empty_cache()
    if report['status'] != 'PASS': raise ValueError(report['error'])
    return report


@torch.no_grad()
def runtime_probe():
    # The baseline checkpoint is permitted ONLY in this runtime-reference scope.
    from src.evaluation.rectangular_formal_evaluation import load_model
    model, checkpoint = load_model()
    before, state_before = parameter_hash(model), state_hash(model)
    raw_maps = []
    for path in normal_paths('validation'):
        x = normal_input(model,path)
        t,s,_,_ = s1.s1_features(model,x)
        raw_maps.append(s1.s1_map(s1.discrepancy(t,s)))
    qa,qb = model._get_quantiles_of_maps(raw_maps)
    del raw_maps,t,s
    x = normal_input(model,normal_paths('train')[0])
    result = dict(status='RUNNING', checkpoint=checkpoint, warmup=20, measured_forwards=200,
                  s1_quantiles_source='19 normal validation, runtime reference only; not P2 fitted quantiles',
                  qa_s1=float(qa), qb_s1=float(qb), baseline_weights_never_used_for_training=True)
    for name,call in [('baseline', lambda: model.model(x)),
                      ('multiscale',lambda: s1.inference_outputs(model,x,qa,qb))]:
        for _ in range(20): output=call()
        del output
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        timings=[]
        for i in range(200):
            torch.cuda.synchronize(); start=time.perf_counter(); output=call(); torch.cuda.synchronize()
            timings.append((time.perf_counter()-start)*1000)
            del output
            if (i+1)%50==0: print(f'Runtime {name}: {i+1}/200',flush=True)
        result[name]=dict(mean_ms=float(np.mean(timings)),median_ms=float(np.median(timings)),
                          std_ms=float(np.std(timings)),samples_ms=timings,
                          peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                          peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
    baseline=result['baseline']['mean_ms']; ms=result['multiscale']['mean_ms']; overhead=(ms/baseline-1)*100
    result.update(overhead_ms=ms-baseline, overhead_percent=overhead,
                  runtime_feasible=True if overhead<=15 else 'CAUTION' if overhead<=25 else False,
                  parameter_state_unchanged=before==parameter_hash(model) and state_before==state_hash(model),
                  runtime_scope='Frozen Rect structural reference, normal train sample; no P2 performance result',
                  order='Baseline then multiscale; single fixed order, timing noise not removed', status='PASS')
    if not result['parameter_state_unchanged']: raise ValueError('FAIL_RUNTIME_STATE_MUTATION')
    save('runtime_probe.json',result)
    del model,x
    gc.collect(); torch.cuda.empty_cache()
    return result
