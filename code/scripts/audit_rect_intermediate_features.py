"""Phase14A structural selection using only normal training images."""
import argparse
import gc
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from src.evaluation import intermediate_feature_audit as a

def stamp():return datetime.now().astimezone().isoformat()

def draw(stages):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(13,9))
    labels=['Input: 3 x 256 x 1024']+[f"{s['stage_id']} | {s['teacher_module']} | {s['C']} x {s['H']} x {s['W']}\nNative RF {s['RF_native'][0]:g} px | {s['cell_ratio_vs_final']:.4f}x final cells | {s.get('candidate_role','')}" for s in stages]
    cells=[256*1024]+[s['cells'] for s in stages]
    colors=['#9ca3af']+['#10b981' if s.get('candidate_selected') else '#3b82f6' for s in stages]
    ax.barh(np.arange(len(cells)),cells,color=colors)
    ax.set_yticks(np.arange(len(cells)),labels);ax.invert_yaxis();ax.set_xlabel('Spatial feature cells (H x W)')
    ax.set_title('Rectangular PDN: actual spatial stages\nStructural selection only; no intermediate detection performance used')
    fig.tight_layout();fig.savefig(a.OUT/'figures/feature_resolution_pyramid.png',dpi=150);plt.close(fig)

def summary(r):
    lines=['='*40,'PHASE 14A - INTERMEDIATE FEATURE AUDIT','='*40,f"Checkpoint: {r.get('strict_load')}",'Input: 256x1024',f"Parameters: {r.get('parameters')}"]
    if 'final' in r:
        s=r['final'];lines+=['-'*40,'FINAL STAGE','Shape: 384 x 56 x 248',f"Cells: {s['cells']}",f"Native stride y/x: {s['stride_native']}",f"Native RF y/x: {s['RF_native']}"]
    lines+=['-'*40,'VALID INTERMEDIATE STAGES']
    for s in r.get('stages',[]):
        if not s['candidate_valid']:continue
        lines += [f"{s['stage_id']}: Teacher={s['teacher_module']}; Student={s['student_module']}",f"Shape: {s['teacher_shape']}; Cell ratio: {s['cell_ratio_vs_final']:.6f}",
            f"Native stride y/x: {s['stride_native']}; Native RF y/x: {s['RF_native']}",f"Gradient peak: {s.get('gradient_probe_peak_vram')} MiB; finite: {s.get('gradient_finite')}; feasible: {s.get('training_feasible')}",f"Runtime overhead: {s.get('runtime_overhead_percent')}%"]
    lines+=['-'*40,'SELECTED CANDIDATES']
    for key in ('candidate_A','candidate_B'):
        s=r.get(key)
        lines.append(key+': '+(f"{s['stage_id']} {s['teacher_module']} | {s['teacher_shape']} | {s['cell_ratio_vs_final']:.6f}x cells | RF_native={s['RF_native']} | peak={s.get('gradient_probe_peak_vram')} MiB" if s else 'NONE'))
    lines+=['-'*40,'SELECTION INTEGRITY','Used anomaly GT: False','Used test AUROC/AUPRO: False','Projector required: False',f"Candidate count: {r.get('candidate_count',0)}",f"Parameter/state unchanged: {r.get('parameters_unchanged')} / {r.get('state_unchanged')}",
            'Intermediate T/S features have not been trained to match; discrepancy values are not layer-quality scores.',a.PROBE_SCOPE,a.RUNTIME_CAVEAT,'STATUS: '+r['status'],r.get('error',''),'='*40]
    return '\n'.join(lines)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--precheck-only',action='store_true');args=parser.parse_args();os.chdir(ROOT)
    if (a.OUT/'candidate_report.json').exists():print('FAIL_EXISTING_OUTPUT: Phase14A report already exists; refusing overwrite',flush=True);return 1
    (a.OUT/'figures').mkdir(parents=True,exist_ok=True)
    r=dict(status='RUNNING',started=stamp(),selection_used_anomaly_gt=False,selection_used_test_metric=False,no_optimizer_steps=True,projector_required=False)
    model=None
    try:
        torch.set_num_threads(4);a.seed();a.protocol_gate();paths=a.train_paths()
        indices=np.random.default_rng(42).choice(len(paths),10,replace=False);selected=[paths[int(i)] for i in indices]
        r.update(normal_train_count=len(paths),inventory_sample_count=10,normal_sample_paths=[str(p) for p in selected],sampling_seed=42,
                 input_integrity={str(p):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,sha256=a.digest(p)) for p in [a.CHECKPOINT,*selected]})
        model,details=a.load_model();r.update(details);r.update(parameters_before=a.parameter_hash(model),state_before=a.state_hash(model))
        arch=a.architecture(model);inventories=[];stages=None
        for i,path in enumerate(selected):
            x=a.input_tensor(model,path);inv=a.inventory(model,x);current=a.make_stages(inv)
            if stages is None:stages=current
            elif current!=stages:raise ValueError('FAIL_NONCONSTANT_STAGE_GEOMETRY')
            inventories.append(dict(source=str(path),**inv));del x
            print(f'Normal inventory {i+1}/10: hooked output unchanged; geometry stable',flush=True)
        r.update(stages=stages,final=stages[-1],architecture=arch)
        if a.parameter_hash(model)!=r['parameters_before'] or a.state_hash(model)!=r['state_before']:raise ValueError('FAIL_INVENTORY_STATE_MUTATION')
        a.write_json(a.OUT/'feature_inventory.json',dict(architecture=arch,samples=inventories,stages=stages,
            stage_definition='Consecutive equal H/W outputs grouped; last Conv/activation preferred. A pool-only distinct H/W is retained explicitly as pool-only, never a fabricated Conv/activation.',
            axes='All two-element geometry fields are y,x. Native scale is4.125 on both axes.'))
        a.write_csv(a.OUT/'feature_inventory.csv',stages)
        if not (a.ROOT/'datasets/imagenette').is_dir():raise ValueError('FAIL_MISSING_IMAGENETTE')
        r['precheck']='PASS'
        print('PRECHECK PASS: SHA / strict load / parameters8058628 / input(256,1024) /10 normal train /all module hooks /RF /no GT or test metrics /no optimizer path',flush=True)
        if args.precheck_only:r['status']='PRECHECK_PASS'
        else:
            x=a.input_tensor(model,selected[0]);extra=a.imagenette_batch(model);r['imagenette_shape']=list(extra.shape)
            probes=[]
            for stage in stages:
                if not stage['candidate_valid']:continue
                probe=a.gradient_probe(model,x,extra,stage);probes.append(probe)
                stage.update(gradient_probe_peak_vram=probe['peak_allocated_mib'],gradient_probe_reserved_vram=probe['peak_reserved_mib'],gradient_finite=probe['gradient_finite'],training_feasible=probe['training_feasible'])
                print(f"Gradient {stage['stage_id']}: peak={probe['peak_allocated_mib']:.4f} MiB, finite={probe['gradient_finite']}, state unchanged",flush=True)
                a.write_json(a.OUT/'gradient_probe.json',dict(sample=str(selected[0]),batch=1,FP32=True,optimizer_steps=0,probes=probes))
            del extra;gc.collect();torch.cuda.empty_cache()
            runtime=a.runtime_probe(model,x,stages);a.write_json(a.OUT/'runtime_probe.json',runtime)
            for stage in stages:
                stage.update(candidate_selected=False,candidate_role='FINAL_REFERENCE' if stage['final_reference'] else '')
                if stage['candidate_valid']:stage['runtime_overhead_percent']=runtime['paths'][stage['stage_id']]['overhead_percent']
            high,mid=a.select_candidates(stages)
            if high is None:raise ValueError('FAIL_NO_FEASIBLE_CANDIDATE')
            for role,s in (('HIGH_RES',high),('MID_RES',mid)):
                if s is not None:s.update(candidate_selected=True,candidate_role=role,role=role)
            r.update(candidate_A=high,candidate_B=mid,candidate_count=1+int(mid is not None),runtime=runtime,
                selection_rule='A: maximum cells, smaller C, deeper. B: nearest geometric midpoint in log cells, excluding same resolution; deterministic tie smaller C then deeper. All require natural full shape match, finite gradient and allocated<=6500MiB.',
                runtime_not_used_to_rank=True)
            if r['candidate_count']>2:raise ValueError('FAIL_TOO_MANY_CANDIDATES')
            a.write_json(a.OUT/'feature_inventory.json',dict(architecture=arch,samples=inventories,stages=stages,stage_definition='Last Conv/activation per exact resolution; pool-only resolution explicitly retained with no fabricated activation.'))
            a.write_csv(a.OUT/'feature_inventory.csv',stages);draw(stages)
            for path,old in r['input_integrity'].items():
                p=Path(path)
                if p.stat().st_size!=old['size'] or p.stat().st_mtime_ns!=old['mtime_ns'] or a.digest(p)!=old['sha256']:raise ValueError('FAIL_FROZEN_INPUT_MUTATION')
            r['status']='PASS'
    except Exception as exc:
        r.update(status='FAIL',error=str(exc),traceback=traceback.format_exc());traceback.print_exc()
    finally:
        if model is not None:
            model.zero_grad(set_to_none=True)
            r.update(parameters_after=a.parameter_hash(model),state_after=a.state_hash(model))
            r.update(parameters_unchanged=r['parameters_before']==r['parameters_after'],state_unchanged=r['state_before']==r['state_after'])
            if not r['parameters_unchanged'] or not r['state_unchanged']:r.update(status='FAIL',error='FAIL_PARAMETER_STATE_MUTATION')
        r['finished']=stamp();stem='precheck_report' if args.precheck_only else 'candidate_report'
        if (a.OUT/(stem+'.json')).exists():stem+='_'+datetime.now().strftime('%Y%m%d_%H%M%S')
        a.write_json(a.OUT/(stem+'.json'),r);(a.OUT/(stem+'.txt')).write_text(summary(r),encoding='utf-8');print(summary(r),flush=True)
    return 0 if r['status'] in ('PASS','PRECHECK_PASS') else 1

if __name__=='__main__':raise SystemExit(main())
