"""Phase13C frozen whole-image evaluation; no fitting or adaptive operation."""
import ast
import gc
import inspect
import shutil
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from anomalib.models import EfficientAd
from anomalib.data.utils import read_image
from src.evaluation import exact_aupro as exact
from src.evaluation import strict_fpr_diagnostics as diagnostic
from src.evaluation.rectangular_efficientad_calibration import TARGET, probe, dataset
from src.evaluation.efficientad_branch_audit import branch_maps, parameter_hash, state_hash
from src.evaluation.st_feature_rootcause import map_from_feature, padding_protocol
from src.evaluation.p1_384_evaluation import read_json, read_csv, write_json, write_csv, digest

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/phase13c_rectangular_256x1024_public'
CHECKPOINT=ROOT/'experiments/phase13b_rectangular_256x1024_seed42/checkpoints/final.ckpt'
SHA='35bc0bb101b97a2cda9e8ab4a97f358c0a8a0447445db640735a1029112415ee'
QUANTILES=dict(qa_st=0.1152532547712326,qb_st=0.18428559601306915,qa_ae=0.04170507937669754,qb_ae=0.14394059777259827)
NATIVE=(1056,4224)
GROUP_COUNTS=dict(A=12,B=18,C=42,D=6,E=12)
METRIC_KEYS=('image_auroc','pixel_auroc','aupro005','aupro030')
P1384=dict(zip(METRIC_KEYS,np.array([84.8148,93.0126,22.0758,45.8642])/100))
B0=ROOT/'experiments/phase07c_baseline256_public'
D0=ROOT/'experiments/phase07d_b0_failure_analysis'
P1=ROOT/'experiments/phase11c_p1_halo96_384_public'
B10=ROOT/'experiments/phase10b_p1_halo96_public'
C10=ROOT/'experiments/phase10c_strict_fpr_diagnostic'
EPS=1e-10

def input_gate(x):
    if tuple(x.shape)!=(1,3,*TARGET):raise ValueError('FAIL_DIMENSION_ORDER')
    if x.dtype!=torch.float32:raise ValueError('FAIL_PRECISION')

def mapping_gate(rows):
    m={r['sample_id']:r for r in rows}
    if len(rows)!=114 or len(m)!=114 or sum(int(r['gt_label'])==0 for r in rows)!=24 or sum(int(r['gt_label'])==1 for r in rows)!=90:
        raise ValueError('FAIL_DATA_MAPPING:114 sources/24 normal/90 anomaly required')
    if len({r['image_path'] for r in rows})!=114:raise ValueError('FAIL_DATA_MAPPING:duplicate source paths')
    return m

def checkpoint_gate(path=CHECKPOINT):
    if digest(path)!=SHA:raise ValueError('FAIL_CHECKPOINT_HASH')
    return SHA

def quantile_gate(q):
    if set(q)!=set(QUANTILES) or any(float(q[k])!=v for k,v in QUANTILES.items()):raise ValueError('FAIL_QUANTILE_STATE')

def preprocessor():
    old=EfficientAd.configure_pre_processor()
    new=EfficientAd.configure_pre_processor(image_size=TARGET)
    if repr(old.transform).replace('[256, 256]','[256, 1024]')!=repr(new.transform):raise ValueError('FAIL_PREPROCESSOR_MISMATCH')
    return new

def load_model():
    import anomalib
    if anomalib.__version__!='2.6.1' or not torch.cuda.is_available():raise ValueError('FAIL_ENVIRONMENT')
    checkpoint_gate();saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    meta=saved.get('phase13b',{});config=meta.get('config',{})
    if saved.get('global_step')!=70000 or not meta.get('final_validation_completed'):raise ValueError('FAIL_CHECKPOINT_PROVENANCE')
    if config.get('input_size_hw')!=[256,1024] or config.get('padding') is not False or config.get('pad_maps') is not True:raise ValueError('FAIL_CHECKPOINT_CONFIG')
    quantile_gate({k:float(saved['state_dict']['model.quantiles.'+k]) for k in QUANTILES});del saved
    model=EfficientAd.load_from_checkpoint(str(CHECKPOINT),strict=True,map_location='cpu',weights_only=False)
    if repr(model.pre_processor.transform)!=repr(preprocessor().transform):raise ValueError('FAIL_PREPROCESSOR_MISMATCH')
    if sum(p.numel() for p in model.parameters())!=8058628:raise ValueError('FAIL_PARAMETERS')
    if not model.model.is_set(model.model.mean_std):raise ValueError('FAIL_TEACHER_STATISTICS')
    q={k:float(v.detach()) for k,v in model.model.quantiles.items()};quantile_gate(q)
    if model.model.ae.decoder.padding is not False or model.model.pad_maps is not True:raise ValueError('FAIL_PADDING')
    model.eval().to('cuda:0')
    return model,dict(checkpoint_sha256=SHA,strict_load=True,parameters=8058628,quantiles=q,preprocessor=repr(model.pre_processor.transform),
        preprocessor_api=str(inspect.signature(EfficientAd.configure_pre_processor)),input_size_hw=list(TARGET))

def input_tensor(model,path):
    with torch.no_grad():
        native=read_image(path,as_tensor=True)
        if tuple(native.shape)!=(3,*NATIVE):raise ValueError('FAIL_NATIVE_SHAPE')
        x=model.pre_processor(native[None].to('cuda:0'));input_gate(x)
    return x

def native_map(raw):
    if tuple(raw.shape)!=(1,1,*TARGET) or raw.dtype!=torch.float32 or not torch.isfinite(raw).all():raise ValueError('FAIL_MAP_SHAPE')
    return torch.nn.functional.interpolate(raw.detach().cpu(),size=NATIVE,mode='bilinear',align_corners=False)[0,0].numpy().copy()

def extract(model,x):
    input_gate(x)
    if model.training or model.model.training:raise ValueError('FAIL_EVAL_MODE')
    with torch.no_grad():
        official=model.model(x).anomaly_map
        student,d=model.model.compute_student_teacher_distance(x)
        if tuple(student.shape)!=(1,768,56,248) or tuple(d.shape)!=(1,384,56,248):raise ValueError('FAIL_SHAPE_GATE')
        st,ae=model.model.compute_maps(x,student,d,normalize=False)
        reconstructed=map_from_feature(d.mean(1,keepdim=True),padding_protocol(model),TARGET)
        branches=branch_maps(st,ae,model.model.quantiles)
        official_st,_=model.model.compute_maps(x,student,d,normalize=True)
        errors=dict(st_raw=float((st-reconstructed).abs().max()),st_normalized=float((official_st-branches['ST_NORMALIZED']).abs().max()),
                    combined=float((official-branches['COMBINED']).abs().max()))
        if any(not np.isfinite(v) or v>1e-5 for v in errors.values()):raise ValueError('FAIL_ST_RECONSTRUCTION: '+str(errors))
    return official.detach(),branches['ST_NORMALIZED'].detach(),errors

def read_membership():
    rows=read_csv(D0/'sample_analysis.csv');m={r['sample_id']:r for r in rows}
    if len(rows)!=90 or len(m)!=90 or {g:sum(r['size_group']==g for r in rows) for g in GROUP_COUNTS}!=GROUP_COUNTS:raise ValueError('FAIL_SIZE_GROUPS')
    return m

def references():
    manifest=read_json(P1/'FROZEN_RESULT_MANIFEST.json')
    if manifest['status']!='PASS' or not manifest['authoritative']:raise ValueError('FAIL_FROZEN_REFERENCE')
    members=read_membership();base=mapping_gate(read_csv(B0/'predictions.csv'));p1=mapping_gate(read_csv(P1/'predictions.csv'))
    if set(base)!=set(p1):raise ValueError('FAIL_DATA_MAPPING')
    dm,counts=dataset();public={str(Path(r.image_path).resolve()):int(r.label_index) for r in dm.test_public_data.samples.itertuples()}
    if public!={str(Path(r['image_path']).resolve()):int(r['gt_label']) for r in base.values()}:raise ValueError('FAIL_PUBLIC_SPLIT_MAPPING')
    expected={(r['sample_id'],int(r['region_id'])):int(r['pixel_count']) for r in read_csv(ROOT/'experiments/phase09a_tile_geometry/region_tile_analysis.csv')}
    if len(expected)!=450:raise ValueError('FAIL_REGIONS')
    files={CHECKPOINT,P1/'FROZEN_RESULT_MANIFEST.json',P1/'sample_metrics.csv',D0/'sample_analysis.csv',B0/'predictions.csv',P1/'predictions.csv',
           C10/'sample_diagnostic.csv',B10/'evaluation_report.json',ROOT/'experiments/phase12b_context_batching/benchmark_report.json',CHECKPOINT.parent.parent/'training_report.json'}
    seen=set()
    for sid,row in base.items():
        row['native_shape']=NATIVE
        if Path(row['image_path']).resolve()!=Path(p1[sid]['image_path']).resolve() or row['gt_label']!=p1[sid]['gt_label']:raise ValueError('FAIL_REFERENCE_MAPPING')
        with Image.open(row['image_path']) as im:
            if im.size!=(4224,1056):raise ValueError('FAIL_NATIVE_SHAPE')
        mask=exact.get_mask(row,Image,np)
        if mask.shape!=NATIVE:raise ValueError('FAIL_GT_SHAPE')
        labels,sizes=exact.components(mask,cv2,np)
        if int(row['gt_label']):
            if int(sizes[1:].sum())!=int(members[sid]['defect_pixel_count']):raise ValueError('FAIL_GT_MAPPING')
            if Path(row['gt_mask_path']).resolve()!=Path(p1[sid]['gt_mask_path']).resolve():raise ValueError('FAIL_GT_MAPPING')
            for rid in range(1,len(sizes)):
                if expected.get((sid,rid))!=int(sizes[rid]):raise ValueError('FAIL_REGION_MAPPING')
                seen.add((sid,rid))
            files.add(Path(row['gt_mask_path']))
        files.add(Path(row['image_path']))
        for path,shape in ((row['raw_map_path'],(256,256)),(p1[sid]['combined_path'],NATIVE),(p1[sid]['st_path'],NATIVE)):
            a=np.load(path,mmap_mode='r',allow_pickle=False)
            if a.shape!=shape or a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('FAIL_REFERENCE_MAP')
            files.add(Path(path))
    if seen!=set(expected):raise ValueError('FAIL_REGION_MAPPING')
    samples={(r['branch'],r['sample_id']):r for r in read_csv(P1/'sample_metrics.csv')}
    b0points={r['sample_id']:r for r in read_csv(C10/'sample_diagnostic.csv') if r['model']=='B0' and int(r['gt_label'])}
    if set(b0points)!=set(members) or any((branch,sid) not in samples for sid in members for branch in ('ST','COMBINED')):raise ValueError('FAIL_PAIRED_MEMBERS')
    tiny=sorted(sid for sid,r in members.items() if r['size_group']=='A')
    worst_b0=sorted(members,key=lambda sid:(float(members[sid]['aupro_005']),sid))[:10]
    worst_p1=sorted(members,key=lambda sid:(float(samples['COMBINED',sid]['per_image_AUPRO005']),sid))[:10]
    group007=[f'{42+i:03d}_007_{c}' for i,c in enumerate(('overexposed','regular','shift_1','shift_2','shift_3','underexposed'))]
    normal=sorted(sid for sid,r in base.items() if not int(r['gt_label']))[:5]
    fixed=dict(tiny=tiny,group007=group007,B0_worst=worst_b0,P1384_worst=worst_p1,normal=normal)
    return dict(mapping=base,p1384=p1,membership=members,tiny=tiny,samples=samples,b0points=b0points,manifest=manifest,
                counts=counts,regions=450,gallery_sets=fixed,gallery_ids=sorted(set(sum(fixed.values(),[]))),
                integrity={str(p):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,sha256=digest(p)) for p in sorted(files)})

def verify_inputs(ref):
    for p,old in ref['integrity'].items():
        path=Path(p)
        if path.stat().st_size!=old['size'] or path.stat().st_mtime_ns!=old['mtime_ns'] or digest(path)!=old['sha256']:raise ValueError('FAIL_FROZEN_INPUT_MUTATION: '+p)

def timed(fn):
    torch.cuda.synchronize();start=time.perf_counter();value=fn();torch.cuda.synchronize()
    return value,(time.perf_counter()-start)*1000

def benchmark_loop(fn,items,rounds,warmup):
    for i in range(warmup):fn(items[i%len(items)])
    times=[]
    for _ in range(rounds):
        for item in items:
            value,ms=timed(lambda:fn(item));times.append(ms);del value
    return times

def runtime_protocol():return dict(warmup=20,model_rounds=5,end_to_end_rounds=3,sources=114)

def stats(values):
    return dict(mean=float(np.mean(values)),median=float(np.median(values)),std=float(np.std(values)),p90=float(np.percentile(values,90)),p95=float(np.percentile(values,95)),fps=1000/float(np.mean(values)),count=len(values))

def benchmark(model,ref):
    paths=[ref['mapping'][sid]['image_path'] for sid in sorted(ref['mapping'])]
    tensors=[input_tensor(model,path).cpu() for path in paths]
    with torch.no_grad():
        for i in range(20):model.model(tensors[i].to('cuda:0'))
        torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        times=[];rounds=[]
        for rnd in range(5):
            local=[]
            for tensor in tensors:
                x=tensor.to('cuda:0');value,ms=timed(lambda:model.model(x));local.append(ms);del value,x
            times.extend(local);rounds.append(stats(local));print(f'Model-only runtime round {rnd+1}/5: {np.mean(local):.4f} ms/source',flush=True)
        memory=dict(peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
        del tensors;gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        def pipeline(path):return native_map(model.model(input_tensor(model,path)).anomaly_map)
        ends=[];endrounds=[]
        for rnd in range(3):
            local=benchmark_loop(pipeline,paths,rounds=1,warmup=0);ends.extend(local);endrounds.append(stats(local))
            print(f'End-to-end runtime round {rnd+1}/3: {np.mean(local):.4f} ms/source',flush=True)
        endmemory=dict(peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
    total=torch.cuda.get_device_properties(0).total_memory/1024**2
    result=dict(protocol=runtime_protocol(),model_only=stats(times),end_to_end=stats(ends),model_rounds=rounds,end_to_end_rounds=endrounds,
                model_only_samples_ms=times,end_to_end_samples_ms=ends,model_memory=memory,end_to_end_memory=endmemory,gpu_total_mib=total,
                peak_allocated_mib=max(memory['peak_allocated_mib'],endmemory['peak_allocated_mib']),peak_reserved_mib=max(memory['peak_reserved_mib'],endmemory['peak_reserved_mib']),
                timing_definition='Synchronized wall-clock official Combined forward only; input H2D excluded. End-to-end includes file read, preprocessing, H2D, forward, map D2H and CPU native bilinear; no metrics/save/plots.',
                comparison_note='P1-384 numbers are frozen Phase12B Batch4 results, not a same-process rerun.')
    result.update(allocated_fraction=result['peak_allocated_mib']/total,latency_reduction_percent=(1-result['model_only']['mean']/196.5630)*100,
                  speedup_x=196.5630/result['model_only']['mean'],end_to_end_reduction_percent=(1-result['end_to_end']['mean']/330.2937)*100,
                  end_to_end_speedup_x=330.2937/result['end_to_end']['mean'])
    return result

def no_training_paths():
    for path in (Path(__file__),ROOT/'scripts/evaluate_rectangular_efficientad_public.py'):
        tree=ast.parse(path.read_text(encoding='utf-8'))
        for n in ast.walk(tree):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in {'fit','train','backward','configure_optimizers','optimizer_step','validate','prepare_imagenette_data'}:raise ValueError('FAIL_TRAINING_PATH')
    return True

def outer_region(mask,width=96):
    y,x=np.nonzero(mask)
    return bool(min(y.min(),x.min(),mask.shape[0]-1-y.max(),mask.shape[1]-1-x.max())<=width)

def metrics(rows,tag,membership,*,output_dir=None):
    # Reuse the exact same frozen metric protocol in later independent phases.
    folder=(Path(output_dir) if output_dir is not None else OUT)/'temp'/tag;meta={}
    exact.native(rows,meta,torch,cv2,np,Image,temp_dir=folder)
    if meta['connected_regions']!=450:raise ValueError('FAIL_REGION_COUNT')
    bg=np.memmap(folder/'background_scores.dat',mode='r',dtype=np.float32,shape=(meta['background_pixels'],))
    pos=np.memmap(folder/'positive_scores.dat',mode='r',dtype=np.float32,shape=(meta['positive_pixels'],))
    weights=np.memmap(folder/'positive_weights.dat',mode='r',dtype=np.float64,shape=(len(pos),))
    auc=sum(float(diagnostic.percentiles(bg,pos[i:i+100000])[2].sum()) for i in range(0,len(pos),100000))/len(pos)/100
    scores=[float(np.load(r['raw_map_path'],mmap_mode='r').max()) for r in rows]
    negatives=np.sort([s for s,r in zip(scores,rows) if not int(r['gt_label'])]);positives=np.array([s for s,r in zip(scores,rows) if int(r['gt_label'])])
    result=dict(image_auroc=float(diagnostic.percentiles(negatives,positives)[2].mean()/100),pixel_auroc=auc,aupro005=meta['aupro_0.05'],aupro030=meta['aupro_0.30'],approximation=False)
    curve=exact.pro_at_fprs(bg,pos,weights,diagnostic.FPRS,np).tolist();samples={};regions=[]
    for index,row in enumerate(rows):
        if not int(row['gt_label']):continue
        sid=row['sample_id'];a=np.load(row['raw_map_path']);labels,sizes=exact.components(exact.get_mask(row,Image,np),cv2,np)
        fg=labels>0;values=a[fg];w=1/((len(sizes)-1)*sizes[labels[fg]].astype(np.float64))
        points=exact.pro_at_fprs(bg,values,w,[.05,.30],np);localbg=np.sort(a[~fg]);areas,_=exact.integrate(localbg,values,w,np)
        samples[sid]=dict(sample_id=sid,branch=tag,size_group=membership[sid]['size_group'],PRO005=float(points[0]),PRO030=float(points[1]),
                          per_image_AUPRO005=areas[.05],per_image_AUPRO030=areas[.30],percentile_p95=float(np.percentile(diagnostic.percentiles(bg,values)[2],95)))
        for rid in range(1,len(sizes)):
            mask=labels==rid;v=a[mask];c,areas=diagnostic.region_pro(bg,v,[.05,.30])
            regions.append(dict(sample_id=sid,region_id=rid,branch=tag,pixels=int(sizes[rid]),outer_boundary=outer_region(mask),
                                PRO005=float(c[0]),PRO030=float(c[1]),AUPRO005=areas[.05],AUPRO030=areas[.30]))
        if (index+1)%15==0:print(f'{tag}: sample/region analysis {index+1}/114',flush=True)
    if any(not 0<=result[k]<=1 for k in METRIC_KEYS):raise ValueError('FAIL_METRIC_RANGE')
    return result,samples,regions,curve,meta

def pareto(metric,latency):
    retention={k:metric[k]/P1384[k] for k in METRIC_KEYS};delta={k:100*(metric[k]-P1384[k]) for k in METRIC_KEYS};saving=1-latency/196.5630
    dominant=all(metric[k]>=P1384[k] for k in METRIC_KEYS[:3]) and saving>0
    strong=not dominant and retention['aupro005']>=.9 and retention['image_auroc']>=.95 and retention['pixel_auroc']>=.95 and saving>=.35
    moderate=retention['aupro005']>=.8 and retention['image_auroc']>=.9 and saving>=.25
    uncompetitive=retention['aupro005']<.8 or retention['image_auroc']<.9 or saving<.15
    flags=dict(RECT_DOMINANT=bool(dominant),RECT_STRONG_PARETO=bool(strong),RECT_MODERATE_PARETO=bool(moderate),RECT_NOT_COMPETITIVE=bool(uncompetitive))
    return dict(**flags,classification=next((k for k,v in flags.items() if v),'UNCLASSIFIED_BY_PREREGISTERED_RULES'),retention=retention,delta_pp=delta,
                reference_precision='User preregistered P1-384 rounded percentages; full precision reference is additionally preserved in Manifest evidence.')
