"""Formal prediction/report orchestration; frozen official model and project metrics."""
import csv,json,time,gc,sys
from pathlib import Path

def finish(model,c,OUT,ROOT,DATA,load,sync,peak,sha,save):
    import torch,numpy as np,cv2
    from PIL import Image
    from sklearn.metrics import roc_auc_score
    sys.path.insert(0,str(ROOT))
    from src.evaluation import exact_aupro as exact
    assert sha(ROOT/'src/evaluation/exact_aupro.py')=='3e6ebc80d400419cf3aaf00950e954ea165df2419c6ea895f4296013c67e11a1'
    # Existing frozen public manifest; verify exact identities against the allowed public folders.
    source=ROOT/'experiments/phase07c_baseline256_public/predictions.csv'
    with source.open(encoding='utf8') as f: frozen=list(csv.DictReader(f))
    actual=set((DATA/'test_public/good').glob('*.png'))|set((DATA/'test_public/bad').glob('*.png'))
    masks=set((DATA/'test_public/ground_truth/bad').glob('*.png'))
    assert len(frozen)==len(actual)==114 and len(masks)==90
    assert {Path(r['image_path']) for r in frozen}==actual
    assert sum(int(r['gt_label']) for r in frozen)==90
    assert {Path(r['gt_mask_path']) for r in frozen if int(r['gt_label'])}==masks
    for r in frozen:
        p=Path(r['image_path']); label=int(r['gt_label'])
        assert label==int(p.parent.name=='bad')
        if label: assert Path(r['gt_mask_path']).name==p.stem+'_mask.png'
    c.update(test_count=114,test_normal_count=24,test_anomaly_count=90,mask_count=90,weight_sha256=c['pretrained_weight_sha256'],pretrained_tag='racm_in1k')
    for sub in ['raw_maps','native_maps','metrics']: (OUT/sub).mkdir(exist_ok=True)
    model.eval(); torch.cuda.reset_peak_memory_stats(); rows=[]; sync(); wall=time.perf_counter()
    with torch.no_grad():
        for i,r in enumerate(frozen):
            p=Path(r['image_path']); sync(); begin=time.perf_counter()
            with Image.open(p) as image: assert image.size==(4224,1056)
            x=load(p); sync(); start=time.perf_counter()
            prediction=model.model(x); sync(); mt=time.perf_counter()-start
            raw=prediction.anomaly_map.detach().float(); score=float(prediction.pred_score.item())
            assert raw.shape==(1,1,256,256) and torch.isfinite(raw).all() and np.isfinite(score)
            start=time.perf_counter()
            native=torch.nn.functional.interpolate(raw,size=(1056,4224),mode='bilinear',align_corners=False)
            sync(); rt=time.perf_counter()-start
            rn=raw[0,0].cpu().numpy(); nn=native[0,0].cpu().numpy()
            assert nn.shape==(1056,4224) and np.isfinite(nn).all()
            sync(); et=time.perf_counter()-begin
            sid=r['sample_id']; rp=OUT/'raw_maps'/f'{sid}.npy'; npth=OUT/'native_maps'/f'{sid}.npy'
            assert not rp.exists() and not npth.exists(), 'Prediction overwrite forbidden'
            np.save(rp,rn); np.save(npth,nn)
            rows.append(dict(sample_id=sid,image_path=str(p),relative_path=str(p.relative_to(DATA)),gt_label=int(r['gt_label']),
              gt_mask_path=r['gt_mask_path'],official_image_score=score,raw_model_map_path=str(rp),raw_map_path=str(npth),
              model_map_shape=[256,256],native_shape=(1056,4224),raw_sha256=sha(rp),native_sha256=sha(npth),
              model_ms=mt*1000,end_to_end_ms=et*1000,native_restore_ms=rt*1000))
            save('prediction_manifest.json',rows)
            if (i+1)%10==0: print('PUBLIC_PREDICTION',i+1,flush=True)
    sync(); iw=time.perf_counter()-wall; ip=peak()
    assert len(rows)==len({r['sample_id'] for r in rows})==114
    c.update(test_prediction_count=114,native_map_count=114,model_only_latency_ms=float(np.mean([r['model_ms'] for r in rows])),
      end_to_end_latency_ms=float(np.mean([r['end_to_end_ms'] for r in rows])),native_restore_latency_ms=float(np.mean([r['native_restore_ms'] for r in rows])),
      inference_wall_time_sec=iw,inference_peak_allocated_mib=ip['allocated_mib'],inference_peak_reserved_mib=ip['reserved_mib'],
      patchcore_image_score_definition='OFFICIAL_PATCHCORE_WEIGHTED_NN_SCORE')
    c['model_only_fps']=1000/c['model_only_latency_ms']
    save('07_test_inference_summary.json',dict(count=114,unique=114,manifest_sha256=sha(OUT/'prediction_manifest.json'),forward_count=114))
    save('08_image_score_protocol.json',dict(definition=c['patchcore_image_score_definition'],detail='Maximum anomalous patch nearest-neighbor distance with 9-neighbor support weighting; official pred_score retained',map_max_substituted=False))
    save('09_native_restore_summary.json',dict(count=114,shape=[1056,4224],dtype='float32',interpolation='bilinear',align_corners=False))
    save('12_runtime_memory.json',dict(c,timing_note='Cold first forward included. E2E read/preprocess/forward/restore/CPU transfer excludes prediction serialization. Wall time includes serialization/hash. No extra test forwards.'))
    # Predict first; metrics only consume persisted frozen predictions.
    shape=(114,1056,4224)
    scores=np.lib.format.open_memmap(OUT/'metrics/native_scores.npy',mode='w+',dtype=np.float32,shape=shape)
    targets=np.lib.format.open_memmap(OUT/'metrics/native_targets.npy',mode='w+',dtype=np.uint8,shape=shape)
    for i,r in enumerate(rows):
        arr=np.load(r['raw_map_path'],allow_pickle=False)
        assert arr.shape==shape[1:] and arr.dtype==np.float32 and np.isfinite(arr).all()
        assert sha(r['raw_map_path'])==r['native_sha256']
        mask=exact.get_mask(r,Image,np); assert mask.shape==shape[1:]
        if r['gt_label']: assert mask.any()
        scores[i]=arr; targets[i]=mask
    scores.flush(); targets.flush()
    c.update(missing_count=0,duplicate_count=0,shape_mismatch_count=0,nan_count=0,inf_count=0,gt_alignment_pass=True)
    save('13_integrity_audit.json',{k:c[k] for k in ['test_prediction_count','native_map_count','missing_count','duplicate_count','shape_mismatch_count','nan_count','inf_count','gt_alignment_pass']})
    c['image_auroc']=float(roc_auc_score([r['gt_label'] for r in rows],[r['official_image_score'] for r in rows]))
    print('PIXEL_AUROC_START',flush=True); t=time.perf_counter()
    c['pixel_auroc']=float(roc_auc_score(targets.reshape(-1),scores.reshape(-1)))
    c['pixel_auroc_time_sec']=time.perf_counter()-t
    del scores,targets; gc.collect()
    save('10_evaluation_report.json',c)
    er={}; t=time.perf_counter()
    exact.native(rows,er,torch,cv2,np,Image,temp_dir=OUT/'metrics/exact_temp')
    c['exact_metrics_combined_time_sec']=time.perf_counter()-t
    c.update(exact_aupro005=er['aupro_0.05'],exact_aupro030=er['aupro_0.30'])
    save('11_aupro_exact_report.json',dict(er,approximation=False,combined_time_sec=c['exact_metrics_combined_time_sec'],individual_timing='Shared exact implementation integrates both limits together; separate times not applicable'))
    read=lambda p:json.loads((ROOT/p).read_text(encoding='utf8'))
    std=read('experiments/phase07c_baseline256_public/evaluation_report.json')['metrics']
    se=read('experiments/phase07c_r1_aupro_exact/aupro_report.json')
    std.update(exact_aupro005=se['aupro_0.05'],exact_aupro030=se['aupro_0.30'])
    pb=read('experiments/phase13c_rectangular_256x1024_public/combined_metrics.json')
    pb.update(exact_aupro005=pb['aupro005'],exact_aupro030=pb['aupro030'])
    keys=['image_auroc','pixel_auroc','exact_aupro005','exact_aupro030']
    comparison={'PatchCore':{k:c[k] for k in keys},'EfficientAD-S':{k:std[k] for k in keys},'PB-ARPR':{k:pb[k] for k in keys},'comparison_type':'method-level, not architecture-controlled'}
    for tag,old in [('efficientad',std),('pb_arpr',pb)]:
        for key,name in zip(keys,['image','pixel','aupro005','aupro030']): c[f'delta_vs_{tag}_{name}_pp']=100*(c[key]-old[key])
    save('14_comparison_with_existing.json',dict(comparison,deltas={k:v for k,v in c.items() if k.startswith('delta_')}))
    table=[dict(method='PatchCore',backbone='wide_resnet50_2.racm_in1k',input=[256,256],training_type='normal-only feature bank + coreset; no gradients',build_train_sec=c['total_build_sec'],model_fps=c['model_only_fps'],peak_vram_mib=c['fit_peak_allocated_mib'],**comparison['PatchCore']),
      dict(method='EfficientAD-S',backbone='official PDN small',input=[256,256],training_type='70K gradient training',**comparison['EfficientAD-S']),
      dict(method='PB-ARPR',backbone='official PDN small',input=[256,1024],training_type='70K gradient training',**comparison['PB-ARPR']),dict(method='FastFlow',status='NOT_RUN')]
    save('15_external_comparison_table.json',table)
    save('10_evaluation_report.json',c)
