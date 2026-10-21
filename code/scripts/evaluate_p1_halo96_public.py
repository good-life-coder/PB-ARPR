"""Frozen Phase10B evaluation. --check-only performs no model forward.

P1 uses native stitched-core maximum for image scoring. Frozen B0/B1 scores
remain their model-output maxima; this provenance difference is explicit.
"""
import argparse
import gc
import hashlib
import inspect
import json
import math
import os
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from src.evaluation import exact_aupro as exact
from src.evaluation.p1_halo_stitch import stitch
from src.evaluation import p1_public_analysis as analysis
from src.data.p1_halo_builder import DERIVED,read_rows,manifest_hash

OUT=ROOT/'experiments/phase10b_p1_halo96_public'
CHECKPOINT=ROOT/'experiments/phase10a_p1_halo96_seed42/checkpoints/final.ckpt'
EXPECTED_SHA256='52b75511be348e8361d5177bb0906c3b76ab60cf7fd990b0279fc5dfbd3980b9'
BASE0=dict(image_auroc=61.57,pixel_auroc=79.54,**{'aupro_0.05':6.41,'aupro_0.30':22.33})
BASE1=dict(image_auroc=56.851851851851855,pixel_auroc=81.35388138552024,
           **{'aupro_0.05':5.270407401461206,'aupro_0.30':24.860098278293247})


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')


def summary(r):
    lines=['='*40,'PHASE 10B - P1 HALO96 PUBLIC EVALUATION','='*40,
        'Dataset: MVTec AD 2 / sheet_metal','Method: P1 Context-Halo EfficientAD-S',
        f"Checkpoint SHA256: {r.get('checkpoint_sha256')}",'Core1056; halo96; context1248; input256;4 contexts/source',
        f"Test sources: {r.get('test_count')}; contexts: {r.get('context_count')}",'-'*40,'OVERALL']
    for key,value in r.get('metrics',{}).items():
        if key in BASE0:lines.append(f'{key}: {value*100:.6f}%; B0:{BASE0[key]}%; B1:{BASE1[key]}%; gain vs B0:{value*100-BASE0[key]:+.6f} pp')
    lines.extend(['-'*40,'SIZE GROUPS'])
    for a in r.get('size_groups',[]):lines.append(json.dumps(a))
    for g,a in r.get('group_paired',{}).items():lines.append(f'Group{g} paired: '+json.dumps(a))
    lines.extend(['ALL ANOMALY PAIRED: '+json.dumps(r.get('all_anomaly_paired',{})),
                  'BOUNDARY: '+json.dumps(r.get('boundary_subgroups',[])),
                  'EFFICIENCY: '+json.dumps(r.get('timing',{})),
                  f"Peak allocated/reserved MiB: {r.get('peak_allocated_mib')} / {r.get('peak_reserved_mib')}",
                  f"Parameters: {r.get('parameters')}; same as B0: {r.get('same_parameters_as_B0')}",
                  f"Training hours: {r.get('training_hours')}; B0:4.2807",
                  'Decision facts only: '+json.dumps(r.get('decision_facts',{})),
                  'Approximation: False','STATUS:',r['status'],str(r.get('error_message') or ''),'='*40])
    return '\n'.join(lines)


def data_mapping(ref,np,Image):
    build=json.loads((DERIVED/'dataset_report.json').read_text(encoding="utf-8"))
    audit=json.loads((ROOT/'experiments/phase09d_p1_halo_audit/p1_halo_audit.json').read_text(encoding="utf-8"))
    digest=manifest_hash()
    if build['status']!='PASS' or not audit['formal_training_ready'] or digest!=build['manifest_sha256'] or digest!=audit['dataset_report']['manifest_sha256']:
        raise ValueError('P1 frozen dataset gate or manifest changed')
    rows=[a for a in read_rows() if a['split']=='test_public']
    if len(rows)!=456 or len({a['context_path'] for a in rows})!=456:raise ValueError('Expected456 unique contexts')
    by_path={str(Path(a['image_path']).resolve()):sid for sid,a in ref['inputs']['B0'].items()}
    groups={sid:[] for sid in ref['inputs']['B0']}
    for a in rows:
        sid=by_path.get(str(Path(a['source_image_path']).resolve()))
        if sid is None:raise ValueError('Unknown P1 source image')
        original=ref['inputs']['B0'][sid];tile=int(a['tile_id'])
        if tile not in (0,1,2,3) or int(a['source_label'])!=int(original['gt_label']):raise ValueError('Source label/core ID mismatch')
        for key,value in (('halo',96),('context_size',1248),('core_size',1056),('core_x0',tile*1056),('core_x1',(tile+1)*1056),('core_y0',0),('core_y1',1056)):
            if int(a[key])!=value:raise ValueError(f'Frozen geometry differs:{key}')
        if json.loads(a['core_offset_in_context'])!=[96,96]:raise ValueError('Core offset mismatch')
        path=Path(a['context_path']).resolve()
        if not path.is_relative_to((DERIVED/'test_public/images').resolve()):raise ValueError('Context outside P1 public data')
        with Image.open(path) as im:
            if im.size!=(1248,1248):raise ValueError('Context must be1248square')
        mp=Path(a['core_mask_path']).resolve()
        if not mp.is_relative_to((DERIVED/'test_public/masks').resolve()):raise ValueError('Core GT outside derived mask directory')
        with Image.open(mp) as im:mask=np.asarray(im)
        if mask.shape!=(1056,1056) or int(np.count_nonzero(mask))!=int(a['core_positive_pixels']):raise ValueError('Core GT metadata mismatch')
        if int(a['tile_is_anomalous'])!=int(bool(mask.any())):raise ValueError('Tile label differs from Core GT')
        groups[sid].append(a)
    for sid,part in groups.items():
        if len(part)!=4 or {int(a['tile_id']) for a in part}!={0,1,2,3}:raise ValueError('Missing/duplicate core IDs')
        part.sort(key=lambda a:int(a['tile_id']))
        if len({a['source_image_id'] for a in part})!=1:raise ValueError('Source ID ambiguity')
        # Exact reassembly of existing masks validates ordering and source GT correspondence.
        tiles=[]
        for a in part:
            with Image.open(a['core_mask_path']) as im:tiles.append(np.asarray(im)>0)
        native=exact.get_mask(ref['inputs']['B0'][sid],Image,np)
        if not np.array_equal(np.concatenate(tiles,axis=1),native):raise ValueError('Core GT/source ordering mismatch')
    if len(groups)!=114 or sum(int(a['gt_label']) for a in ref['inputs']['B0'].values())!=90:raise ValueError('Source counts must be114/24/90')
    return groups,digest


def load_model(report,manifest_digest,torch):
    from anomalib.models import EfficientAd
    saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    meta=saved.get('phase10a',{});config=meta.get('config',{})
    if saved.get('global_step')!=70000 or not meta.get('final_validation_completed'):raise ValueError('Missing final-step/final-validation provenance')
    expected={'halo':96,'core_size':[1056,1056],'context_size':[1248,1248],'model_input':[256,256],
              'tiles_per_source':4,'model_size':'small','seed':42,'manifest_sha256':manifest_digest,
              'post_processor':False,'evaluator':False,'visualizer':False,'trainer_precision':'32-true'}
    for key,value in expected.items():
        if config.get(key)!=value:raise ValueError(f'Frozen checkpoint config differs:{key}')
    state=saved['state_dict'];quantiles={k:float(state['model.quantiles.'+k]) for k in ('qa_st','qb_st','qa_ae','qb_ae')}
    if not all(math.isfinite(v) for v in quantiles.values()) or quantiles['qa_st']>=quantiles['qb_st'] or quantiles['qa_ae']>=quantiles['qb_ae']:
        raise ValueError('Invalid final quantiles')
    del saved,state;gc.collect()
    report['official_load_api']=str(inspect.signature(EfficientAd.load_from_checkpoint))
    model=EfficientAd.load_from_checkpoint(str(CHECKPOINT),map_location='cpu',strict=True,weights_only=False,
        model_size='small',imagenet_dir='./datasets/imagenette',evaluator=False,visualizer=False,post_processor=False)
    if sum(p.numel() for p in model.parameters())!=8058628:raise ValueError('Parameter count differs from frozen B0/P1')
    if repr(model.pre_processor.transform)!=config['preprocessor_configuration']:raise ValueError('Preprocessor differs from final checkpoint')
    if list(model.pre_processor.transform.transforms[0].size)!=[256,256] or model.model.ae.decoder.padding is not False or model.model.pad_maps is not True:
        raise ValueError('Default model/input geometry changed')
    for k,v in quantiles.items():
        if float(model.model.quantiles[k].detach())!=v:raise ValueError('Official full load quantile mismatch')
    model.requires_grad_(False);model.eval()
    report.update(parameters=8058628,same_parameters_as_B0=True,preprocessor=repr(model.pre_processor.transform),
                  frozen_quantiles=quantiles,full_checkpoint_loaded=True)
    return model


def forward_context(model,path,torch):
    from anomalib.data.utils import read_image
    image=read_image(path,as_tensor=True)[None].to('cuda:0')
    if tuple(image.shape[-2:])!=(1248,1248):raise ValueError('Unexpected context size')
    image=model.pre_processor(image)
    if tuple(image.shape[-2:])!=(256,256):raise ValueError('Unexpected model input')
    with torch.inference_mode():
        torch.cuda.synchronize();start=time.perf_counter();prediction=model.model(image);torch.cuda.synchronize()
        elapsed=(time.perf_counter()-start)*1000
        raw=prediction.anomaly_map.detach().float().cpu().squeeze().numpy().copy()
        pred=float(prediction.pred_score.detach().cpu())
    return raw,pred,elapsed


def infer(model,groups,ref,report,np,torch):
    model.to('cuda:0');contexts=[a for sid in sorted(groups) for a in groups[sid]]
    for a in contexts[:10]:forward_context(model,a['context_path'],torch)
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    rows=[];context_rows=[];all_times=[]
    for index,sid in enumerate(sorted(groups)):
        torch.cuda.synchronize();start=time.perf_counter();outputs=[];times=[];preds=[]
        for tile in groups[sid]:
            raw,pred,ms=forward_context(model,tile['context_path'],torch)
            if raw.shape!=(256,256) or raw.dtype!=np.float32 or not np.isfinite(raw).all() or not math.isfinite(pred):raise ValueError('Invalid context prediction')
            outputs.append((int(tile['tile_id']),raw));times.append(ms);preds.append(pred)
        full=stitch(outputs)
        torch.cuda.synchronize();e2e=(time.perf_counter()-start)*1000
        if full.shape!=(1056,4224) or full.dtype!=np.float32 or not np.isfinite(full).all():raise ValueError('Invalid native Core map')
        # All disk writes occur after the end-to-end timing boundary.
        for (tile,raw),ms,pred in zip(outputs,times,preds):
            dest=OUT/'raw_context_maps'/f'{sid}__core{tile}.npy';np.save(dest,raw,allow_pickle=False)
            context_rows.append(dict(sample_id=sid,tile_id=tile,raw_map_path=str(dest),anomalib_pred_score=pred,
                raw_map_max=float(raw.max()),model_only_ms=ms))
        dest=OUT/'raw_native_maps'/(sid+'.npy');np.save(dest,full,allow_pickle=False)
        original=ref['inputs']['B0'][sid]
        row=dict(sample_id=sid,image_path=original['image_path'],gt_label=int(original['gt_label']),gt_mask_path=original['gt_mask_path'],
            native_shape=(1056,4224),native_gt_shape='[1056,4224]',raw_map_path=str(dest),official_image_score=float(full.max()),
            anomaly_map_min=float(full.min()),anomaly_map_max=float(full.max()),anomaly_map_mean=float(full.mean()),
            anomaly_map_shape='[1056,4224]',four_context_model_only_ms=sum(times),end_to_end_ms=e2e)
        rows.append(row);all_times.extend(times)
        analysis.csv_save(OUT/'predictions.csv',rows);analysis.csv_save(OUT/'context_predictions.csv',context_rows)
        report.update(predictions_saved=len(rows),context_maps_saved=len(context_rows))
        if (index+1)%10==0:print(f'P1 native sources saved:{index+1}/114',flush=True)
    if len(list((OUT/'raw_native_maps').glob('*.npy')))!=114 or len(list((OUT/'raw_context_maps').glob('*.npy')))!=456:raise ValueError('Saved map count mismatch')
    def stats(v):
        v=np.asarray(v,dtype=np.float64)
        return dict(n=len(v),mean_ms=float(v.mean()),median_ms=float(np.median(v)),std_ms=float(v.std()))
    report['timing']=dict(context=stats(all_times),source_model_only=stats([a['four_context_model_only_ms'] for a in rows]),
        source_end_to_end=stats([a['end_to_end_ms'] for a in rows]),warmup_context_forwards=10)
    report['timing']['source_fps']=1000/report['timing']['source_model_only']['mean_ms']
    report.update(peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
    model.cpu();torch.cuda.empty_cache()
    return rows


def native_metrics(rows,ref,report,np,torch,cv2,Image):
    from sklearn.metrics import roc_auc_score
    metrics=dict(image_auroc=float(roc_auc_score([a['gt_label'] for a in rows],[a['official_image_score'] for a in rows])))
    report['metrics']=metrics;write_json(OUT/'metrics/public_metrics.json',metrics)
    native_report={};temp=OUT/'metrics/exact_temp'
    exact.native(rows,native_report,torch,cv2,np,Image,temp_dir=temp)
    report['exact_aupro']=native_report
    if native_report['connected_regions']!=450:raise ValueError('Expected450 connected regions')
    bg=np.memmap(temp/'background_scores.dat',mode='r',dtype=np.float32,shape=(native_report['background_pixels'],))
    pos=np.memmap(temp/'positive_scores.dat',mode='r',dtype=np.float32,shape=(native_report['positive_pixels'],))
    weights=np.memmap(temp/'positive_weights.dat',mode='r',dtype=np.float64,shape=(len(pos),))
    # Exact Mann-Whitney form of Pixel AUROC: all native positive pixels vs all native background pixels.
    # Shared Phase08D midrank handles every tie, with no histogram, sigmoid or sampling.
    rank_sum=0.
    for i in range(0,len(pos),250000):rank_sum+=float(analysis.midrank(bg,pos[i:i+250000],np).sum(dtype=np.float64))
    metrics.update(pixel_auroc=rank_sum/len(pos),**{k:native_report[k] for k in ('aupro_0.05','aupro_0.30')})
    if not all(math.isfinite(v) and 0<=v<=1 for v in metrics.values()):raise ValueError('Metric range invalid')
    curve=exact.pro_at_fprs(bg,pos,weights,analysis.FPRS,np)
    curves=[dict(FPR=f,B0_PRO=float(old['B0_PRO']),B1_PRO=float(old['B1_PRO']),P1_PRO=float(v),
                 P1_minus_B0=float(v)-float(old['B0_PRO'])) for f,old,v in zip(analysis.FPRS,ref['curves'],curve)]
    analysis.csv_save(OUT/'metrics/pro_fpr_b0_b1_p1.csv',curves)
    report['strict_fpr_facts']=[a for a in curves if a['FPR']<=.05]
    report['pixel_auroc_implementation']='Exact all-pixel Mann-Whitney midrank; mathematically identical to native ROC AUROC, CPU chunked; shared Phase08D midrank.'
    write_json(OUT/'metrics/public_metrics.json',dict(metrics,percent={k:v*100 for k,v in metrics.items()},test_count=114,normal_count=24,anomaly_count=90,approximation=False))
    del bg,pos,weights;gc.collect()


def comparisons(report):
    rows=[]
    for key,b0 in BASE0.items():rows.append(dict(Metric=key,B0=b0,B1=BASE1[key],P1=report['metrics'][key]*100,
        P1_minus_B0=report['metrics'][key]*100-b0,unit='percent; gain in pp'))
    for a in report['size_groups']:rows.append(dict(Metric='Group'+a['group']+' mean AU-PRO005',B0=a['B0_mean_percent'],
        B1=a['B1_mean_percent'],P1=a['P1_mean_percent'],P1_minus_B0=a['P1_gain_vs_B0_pp'],unit='percent; gain in pp'))
    ms=report['timing']['source_model_only']['mean_ms']
    for label,b0,b1,p1,unit in (('model_only_ms/source',43.1369,74.66787211510214,ms,'ms'),
        ('forwards/source',1,1,4,'count'),('source_FPS',1000/43.1369,1000/74.66787211510214,1000/ms,'FPS'),
        ('parameters',8058628,8058628,8058628,'count'),('training_hours',4.2807,8.0460,report['training_hours'],'hours'),
        ('training_peak_allocated',1185.2217,1550.7520,report['training_peak_allocated_mib'],'MiB'),
        ('inference_peak_allocated',873.2402,None,report['peak_allocated_mib'],'MiB')):
        rows.append(dict(Metric=label,B0=b0,B1=b1,P1=p1,P1_minus_B0=p1-b0,unit=unit))
    analysis.csv_save(OUT/'metrics/B0_B1_P1.csv',rows)
    sub={a['group']:a for a in report['boundary_subgroups']}
    report['decision_facts']=dict(overall_aupro005_gain_vs_B0_pp=report['metrics']['aupro_0.05']*100-6.41,
        overall_aupro030_gain_vs_B0_pp=report['metrics']['aupro_0.30']*100-22.33,
        image_auroc_gain_vs_B0_pp=report['metrics']['image_auroc']*100-61.57,
        pixel_auroc_gain_vs_B0_pp=report['metrics']['pixel_auroc']*100-79.54,
        tiny_mean_gain_vs_B0_pp=report['size_groups'][0]['P1_gain_vs_B0_pp'],
        tiny_improved_count=report['group_paired']['A']['improved_count'],tiny_worsened_count=report['group_paired']['A']['worsened_count'],
        all_anomaly_median_paired_gain_pp=report['all_anomaly_paired']['median_delta_pp'],
        boundary_group_gain=sub['outer_boundary_le96']['mean_rank_gain_pp'],interior_group_gain=sub['interior_gt96']['mean_rank_gain_pp'],
        boundary_gain_unit='background percentile rank percentage points, not AU-PRO pp',
        inference_time_ratio_vs_B0=ms/43.1369,training_time_ratio_vs_B0=report['training_hours']/4.2807)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    report=dict(status='NOT_RUN',error_message=None,timestamp=datetime.now().astimezone().isoformat(),approximation=False,
        checkpoint_path=str(CHECKPOINT),expected_sha256=EXPECTED_SHA256,seed=42,core=1056,halo=96,context=1248,input_size=[256,256],
        image_score_rule='max of native stitched continuous Core map',
        baseline_score_provenance='Frozen B0/B1 use max of continuous model-output map before native resize. P1 follows requested native Core maximum; original baseline results not redefined.',
        timing_policy='10 separate warmup context forwards; all114 sources measured afterwards. Synchronized model-only sums4 forwards. End-to-end includes read/preprocess/transfer/forward/CPU bilinear1248/crop/stitch, excludes artifact writes.',
        checkpoint_modified=False,no_threshold_tuning=True,no_adaptation=True)
    stage='FAIL_OTHER';start=time.perf_counter();preserve=False
    try:
        os.chdir(ROOT);os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
        for directory in ('','raw_context_maps','raw_native_maps','metrics','visualizations','paired_analysis'):(OUT/directory).mkdir(parents=True,exist_ok=True)
        if not args.check_only and (OUT/'predictions.csv').exists():
            preserve=True;raise FileExistsError('Existing predictions preserved; refusing overwrite/repeated inference')
        stage='FAIL_CHECKPOINT_HASH'
        report['checkpoint_sha256']=sha(CHECKPOINT)
        if report['checkpoint_sha256']!=EXPECTED_SHA256:raise ValueError('Checkpoint SHA256 mismatch; evaluation stopped')
        import numpy as np
        import torch
        import torchvision
        import cv2
        from PIL import Image
        from lightning.pytorch import seed_everything
        from importlib.metadata import version
        report.update(python_version=platform.python_version(),torch_version=str(torch.__version__),torchvision_version=torchvision.__version__,
            cuda_runtime=torch.version.cuda,anomalib_version=version('anomalib'),lightning_version=version('lightning'))
        if report['anomalib_version']!='2.6.1':raise ValueError('Requires frozen Anomalib2.6.1')
        seed_everything(42,workers=True)
        stage='FAIL_DATA_MAPPING';ref=analysis.references(np,Image);groups,digest=data_mapping(ref,np,Image)
        report.update(test_count=114,context_count=456,normal_count=24,anomaly_count=90,manifest_sha256=digest,
            sample_mapping='PASS',size_group_counts=analysis.GROUP_N,fixed_gallery_selection=ref['galleries'],reference_hashes=ref['hashes'])
        stage='FAIL_CHECKPOINT_LOAD';model=load_model(report,digest,torch)
        frozen=json.loads((ROOT/'experiments/phase10a_p1_halo96_seed42/training_report.json').read_text(encoding="utf-8"))
        if frozen['status']!='PASS' or frozen['checkpoint_sha256']!=EXPECTED_SHA256:raise ValueError('Formal training report mismatch')
        report.update(training_hours=frozen['training_elapsed_hours'],training_peak_allocated_mib=frozen['peak_gpu_allocated_mib'])
        # Tiny CPU tie control validates shared rank AUROC equivalence, with no P1 score use.
        from sklearn.metrics import roc_auc_score
        bg=np.array([0,0,1,2],dtype=np.float32);pos=np.array([0,1,1,3],dtype=np.float32)
        delta=abs(float(analysis.midrank(bg,pos,np).mean())-roc_auc_score([0]*4+[1]*4,np.r_[bg,pos]))
        if delta>1e-12:raise ValueError('Exact Pixel AUROC tie-control failed')
        report['pixel_auroc_tie_control_difference']=delta
        if args.check_only:
            report['status']='READY_NOT_EVALUATED';return 0
        stage='FAIL_INFERENCE'
        if not torch.cuda.is_available():raise RuntimeError('GPU0 required')
        torch.cuda.set_device(0);report.update(gpu_name=torch.cuda.get_device_name(0),gpu_total_mib=torch.cuda.get_device_properties(0).total_memory/1024**2)
        import subprocess
        driver=subprocess.run(['nvidia-smi','--query-gpu=index,name,driver_version','--format=csv'],capture_output=True,text=True,timeout=20,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),check=True)
        report['driver_info']=driver.stdout.strip()
        rows=infer(model,groups,ref,report,np,torch);del model;gc.collect()
        stage='FAIL_METRIC';native_metrics(rows,ref,report,np,torch,cv2,Image)
        write_json(OUT/'evaluation_report.json',report)
        stage='FAIL_ANALYSIS';analysis.analyze(rows,ref,OUT,report,np,torch,cv2,Image);comparisons(report)
        write_json(OUT/'evaluation_report.json',report)
        stage='FAIL_VISUALIZATION';analysis.figures_and_galleries(rows,ref,OUT,report,np,torch,Image)
        stage='FAIL_INPUT_CHANGED'
        if sha(CHECKPOINT)!=EXPECTED_SHA256 or manifest_hash()!=digest or any(sha(p)!=h for p,h in ref['hashes'].items()):raise ValueError('Frozen input changed during evaluation')
        report['status']='PASS';return 0
    except (Exception,KeyboardInterrupt) as exc:
        report.update(status=stage,error_message=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());traceback.print_exc();return 1
    finally:
        report['elapsed_seconds']=time.perf_counter()-start;OUT.mkdir(parents=True,exist_ok=True)
        name='preflight_report' if args.check_only else 'refused_run' if preserve else 'evaluation_report'
        write_json(OUT/(name+'.json'),report);(OUT/(name+'.txt')).write_text(summary(report),encoding='utf-8');print(summary(report))


if __name__=='__main__':raise SystemExit(main())
