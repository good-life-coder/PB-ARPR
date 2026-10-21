"""Phase11C frozen evaluation gates and shared native analysis orchestration.

PRO denotes a point on the global-background curve; AU-PRO denotes its integral.
H13 uses the frozen Phase10D PRO point definition, not local per-image AU-PRO.
"""
import csv
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from src.evaluation import exact_aupro as exact
from src.evaluation import strict_fpr_diagnostics as diagnostic
from src.evaluation import st_feature_rootcause as feature
from src.evaluation.efficientad_branch_audit import branch_maps,parameter_hash,state_hash

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/phase11c_p1_halo96_384_public'
CHECKPOINT=ROOT/'experiments/phase11b_p1_halo96_384_seed42/checkpoints/final.ckpt'
SHA='2db70dfef9500d974057b7c298ed885fd3e8e9d313e4703aaf2a85bb72fae8eb'
D10=ROOT/'experiments/phase10d_branch_decomposition'
E10=ROOT/'experiments/phase10e_st_feature_rootcause'
B10=ROOT/'experiments/phase10b_p1_halo96_public'
GROUP_COUNTS=dict(A=12,B=18,C=42,D=6,E=12)
EPS=1e-10

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def read_csv(path):
    with Path(path).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def write_json(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def checkpoint_gate(path=CHECKPOINT):
    if digest(path)!=SHA:raise ValueError('FAIL_CHECKPOINT_HASH')
    return SHA
def pair_gate(a,b):
    if set(a)!=set(b):raise ValueError('FAIL_DATA_MAPPING: sample IDs differ')
def mapping_gate(groups):
    if len(groups)!=114 or any(len(v)!=4 or [int(r['tile_id']) for r in v]!=[0,1,2,3] for v in groups.values()):
        raise ValueError('FAIL_DATA_MAPPING: expected114 sources /456 ordered contexts')
def shape_gate(teacher,student,discrepancy):
    if tuple(teacher)!=(1,384,88,88) or tuple(student)!=(1,768,88,88) or tuple(discrepancy)!=(1,384,88,88):
        raise ValueError('FAIL_FEATURE_GEOMETRY')
def quantile_gate(q):
    if set(q)!={'qa_st','qb_st','qa_ae','qb_ae'} or not all(np.isfinite(float(v)) for v in q.values()):raise ValueError('FAIL_QUANTILE_STATE')
    if not q['qa_st']<q['qb_st'] or not q['qa_ae']<q['qb_ae']:raise ValueError('FAIL_QUANTILE_STATE')
def preprocessor():
    from anomalib.models import EfficientAd
    old=EfficientAd.configure_pre_processor(image_size=(256,256))
    new=EfficientAd.configure_pre_processor(image_size=(384,384))
    if repr(old.transform).replace('[256, 256]','[384, 384]')!=repr(new.transform):raise ValueError('FAIL_PREPROCESSOR_MISMATCH')
    return new
def core_map(raw):
    raw=np.asarray(raw)
    if raw.shape!=(384,384) or raw.dtype!=np.float32 or not np.isfinite(raw).all():raise ValueError('Invalid384 continuous map')
    return torch.nn.functional.interpolate(torch.from_numpy(raw)[None,None],(1248,1248),mode='bilinear',align_corners=False)[0,0,96:1152,96:1152].numpy().copy()
def stitch(maps):
    if len(maps)!=4:raise ValueError('Four ordered cores required')
    return np.concatenate([core_map(m) for m in maps],axis=1)
def frozen_references():
    report=read_json(D10/'branch_report.json');features=read_json(E10/'feature_report.json')
    if report['status']!='PASS' or features['status']!='PASS':raise ValueError('Frozen diagnostic references not PASS')
    tiny={r['sample_id']:r for r in read_csv(E10/'tiny_feature_diagnostic.csv')}
    samples={(r['branch'],r['sample_id']):r for r in read_csv(D10/'sample_branch_metrics.csv')}
    if len(tiny)!=12:raise ValueError('Tiny membership changed')
    return dict(report=report,features=features,tiny=tiny,samples=samples)
def references():
    from scripts.diagnose_p1_strict_fpr import precheck,GROUP007
    from scripts.evaluate_p1_halo96_public import data_mapping
    mapping,membership,tiny,hashes,counts=precheck()
    r=frozen_references();pair_gate(tiny,r['tiny'])
    groups,manifest=data_mapping({'inputs':{'B0':mapping['B0']}},np,Image);mapping_gate(groups)
    if {g:sum(a['size_group']==g for a in membership.values()) for g in GROUP_COUNTS}!=GROUP_COUNTS:raise ValueError('Size group mismatch')
    old_st={sid:dict(row,raw_map_path=str(D10/'raw_maps/st_normalized'/f'{sid}.npy')) for sid,row in mapping['P1'].items()}
    if len(list((D10/'raw_maps/st_normalized').glob('*.npy')))!=114:raise ValueError('Frozen ST maps missing')
    for row in old_st.values():
        p=Path(row['raw_map_path']);a=np.load(p,mmap_mode='r',allow_pickle=False)
        if a.shape!=(1056,4224) or a.dtype!=np.float32:raise ValueError('Frozen ST geometry mismatch')
        hashes[str(p)]=digest(p)
    files=[D10/'branch_report.json',D10/'sample_branch_metrics.csv',E10/'tiny_feature_diagnostic.csv',E10/'feature_report.json',E10/'feature_geometry.json',B10/'metrics/pro_fpr_b0_b1_p1.csv']
    for p in files:hashes[str(p)]=digest(p)
    worst=sorted(membership,key=lambda sid:(float(r['samples'][('COMBINED',sid)]['per_image_AUPRO005']),sid))[:10]
    r.update(mapping=mapping['P1'],membership=membership,groups=groups,manifest=manifest,hashes=hashes,counts=counts,old_st=old_st,gallery_ids=sorted(set(tiny)|set(GROUP007)|set(worst)))
    return r
def load_model(manifest):
    from anomalib.models import EfficientAd
    import anomalib
    if anomalib.__version__!='2.6.1':raise ValueError('Anomalib version mismatch')
    checkpoint_gate();saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    meta=saved.get('phase11b',{});c=meta.get('config',{})
    if saved.get('global_step')!=70000 or not meta.get('final_validation_completed'):raise ValueError('Final step/validation provenance missing')
    expected=dict(core_size=[1056,1056],halo=96,context_size=[1248,1248],model_input=[384,384],tiles_per_source=4,model_size='small',seed=42,trainer_precision='32-true',manifest_sha256=manifest)
    for k,v in expected.items():
        if c.get(k)!=v:raise ValueError(f'Frozen checkpoint config mismatch:{k}')
    q={k:float(saved['state_dict']['model.quantiles.'+k]) for k in ('qa_st','qb_st','qa_ae','qb_ae')};quantile_gate(q)
    del saved
    model=EfficientAd.load_from_checkpoint(str(CHECKPOINT),strict=True,weights_only=False,map_location='cpu',model_size='small',imagenet_dir='./datasets/imagenette',pre_processor=preprocessor(),evaluator=False,post_processor=False,visualizer=False)
    if sum(p.numel() for p in model.parameters())!=8058628:raise ValueError('FAIL_PARAMETERS')
    if repr(model.pre_processor.transform)!=c['preprocessor_configuration']:raise ValueError('Checkpoint preprocessing mismatch')
    if not model.model.is_set(model.model.mean_std):raise ValueError('Teacher statistics missing')
    loaded_q={k:float(v.detach()) for k,v in model.model.quantiles.items()}
    quantile_gate(loaded_q)
    if loaded_q!=q:raise ValueError('Full load changed frozen quantiles')
    model.requires_grad_(False);model.eval();model.to('cuda:0')
    return model,dict(checkpoint_sha256=SHA,strict_load=True,parameters=8058628,quantiles=q,preprocessor=repr(model.pre_processor.transform),preprocessor_api=str(inspect.signature(EfficientAd.configure_pre_processor)),only_resize_changed=True)
def input_tensor(model,path):
    from anomalib.data.utils import read_image
    x=read_image(path,as_tensor=True)[None].to('cuda:0')
    if tuple(x.shape[-2:])!=(1248,1248):raise ValueError('Context mismatch')
    x=model.pre_processor(x)
    if tuple(x.shape[-2:])!=(384,384):raise ValueError('Input mismatch')
    return x
def extract(model,x,geometry_audit=False):
    captures={};layers=[];handles=[]
    def teacher_capture(module,args,out):captures['teacher']=out.detach()
    def spatial_capture(module,args,out):
        layers.append(dict(kind=type(module).__name__,kernel=feature.pair(module.kernel_size),stride=feature.pair(module.stride or module.kernel_size),padding=feature.pair(module.padding),dilation=feature.pair(getattr(module,'dilation',1)),ceil_mode=bool(getattr(module,'ceil_mode',False)),observed_output_shape=list(out.shape[-2:])))
    if geometry_audit:
        handles.append(model.model.teacher.register_forward_hook(teacher_capture))
        for m in model.model.teacher.modules():
            if isinstance(m,(torch.nn.Conv2d,torch.nn.AvgPool2d,torch.nn.MaxPool2d)):handles.append(m.register_forward_hook(spatial_capture))
    try:
        with torch.no_grad():
            student,d=model.model.compute_student_teacher_distance(x)
            shape_gate((1,model.model.teacher_out_channels,*d.shape[-2:]),student.shape,d.shape)
            if geometry_audit:
                shape_gate(captures['teacher'].shape,student.shape,d.shape)
                expected=((captures['teacher']-model.model.mean_std['mean'])/model.model.mean_std['std']-student[:,:model.model.teacher_out_channels])**2
                if float((expected-d).abs().max())>1e-5:raise ValueError('Discrepancy formula mismatch')
            st,ae=model.model.compute_maps(x,student,d,normalize=False)
            reconstructed=feature.map_from_feature(d.mean(1,keepdim=True),feature.padding_protocol(model),x.shape[-2:])
            maps=branch_maps(st,ae,model.model.quantiles)
            sn,_=model.model.compute_maps(x,student,d,normalize=True)
            err=max(float((reconstructed-st).abs().max()),float((sn-maps['ST_NORMALIZED']).abs().max()))
            if err>1e-5:raise ValueError('FAIL_ST_RECONSTRUCTION')
            g=feature.geometry(layers,(384,384)) if geometry_audit else None
            if g and (g['output_shape']!=[88,88] or g['rf']!=[33.,33.] or g['jump']!=[4.,4.]):raise ValueError('RF geometry mismatch')
            return maps['ST_NORMALIZED'][0,0].cpu().numpy().copy(),maps['COMBINED'][0,0].cpu().numpy().copy(),err,g
    finally:
        for h in handles:h.remove()
def h13(old,new,occ_old,occ_new,overall,baseline,p95):
    old=np.asarray(old);new=np.asarray(new);delta=new-old;improved=int((delta>EPS).sum())
    median=float(np.median(delta));mean=float(delta.mean());oldocc=float(np.median(occ_old));newocc=float(np.median(occ_new))
    conditions=[improved>=7,median>EPS,bool(((old<=EPS)&(new>EPS)).any()),newocc>=1.25*oldocc and newocc>oldocc+EPS,overall>=baseline-1e-6]
    count=sum(conditions)
    if count==5:status='STRONGLY_SUPPORTED'
    elif count==4 and improved>=6:status='SUPPORTED'
    elif improved<=2 and abs(median)<=EPS and float(new.mean())<=EPS:status='NOT_SUPPORTED'
    elif count in (2,3) or 3<=improved<=5 or (mean>EPS and abs(median)<=EPS):status='PARTIALLY_SUPPORTED'
    else:status='INCONCLUSIVE'
    return dict(status=status,conditions={f'condition_{i+1}':bool(c) for i,c in enumerate(conditions)},improved_count=improved,worsened_count=int((delta<-EPS).sum()),unchanged_count=int((abs(delta)<=EPS).sum()),mean_delta_pp=mean*100,median_delta_pp=median*100,median_occupancy_256=oldocc,median_occupancy_384=newocc,H15_strengthened=bool(conditions[3] and np.all(new<=EPS) and (np.asarray(p95)<90).sum()>=7),rules='C4 >=25% median occupancy increase fixed before results; PRO epsilon1e-10, overall AU-PRO tolerance1e-6; explicit NOT_SUPPORTED precedes geometry-only partial support')
def tiny_occupancy(row,rid,g):
    import cv2
    labels,sizes=exact.components(exact.get_mask(row,Image,np),cv2,np);region=labels==rid;occ=[];pixels=0
    for i in range(4):
        mask=feature.context_mask(region,i);pixels+=int(mask.sum())
        if mask.any():occ.append(feature.rf_occupancy(feature.fractional_mask(mask,size=(384,384)),g)[0].ravel())
    if pixels!=int(sizes[rid]):raise ValueError('Cross-core GT partition failed')
    a=np.concatenate(occ);maximum=float(a.max())
    return dict(region_id=rid,region_pixels=pixels,max_rf_occupancy=maximum,effective_feature_cells=float(a.sum()/maximum) if maximum else 0.,cells_occupancy_ge025=int((a>=.25).sum()),cells_occupancy_ge050=int((a>=.5).sum()))

def metrics(rows,tag,membership):
    """Identical R1 memmaps/integral; exact pixel AUROC uses tie-midrank."""
    import cv2
    meta={};folder=OUT/'temp'/tag
    exact.native(rows,meta,torch,cv2,np,Image,temp_dir=folder)
    bg=np.memmap(folder/'background_scores.dat',mode='r',dtype=np.float32,shape=(meta['background_pixels'],))
    pos=np.memmap(folder/'positive_scores.dat',mode='r',dtype=np.float32,shape=(meta['positive_pixels'],))
    weights=np.memmap(folder/'positive_weights.dat',mode='r',dtype=np.float64,shape=(len(pos),))
    pixel=sum(float(diagnostic.percentiles(bg,pos[i:i+100000])[2].sum()) for i in range(0,len(pos),100000))/len(pos)/100
    curve=exact.pro_at_fprs(bg,pos,weights,diagnostic.FPRS,np).tolist()
    scores=[float(np.load(r['raw_map_path'],mmap_mode='r').max()) for r in rows]
    negatives=np.sort([s for s,r in zip(scores,rows) if not int(r['gt_label'])]);positives=np.array([s for s,r in zip(scores,rows) if int(r['gt_label'])])
    result=dict(image_auroc=float(diagnostic.percentiles(negatives,positives)[2].mean()/100),pixel_auroc=pixel,aupro005=meta['aupro_0.05'],aupro030=meta['aupro_0.30'],approximation=False,test_count=114,normal_count=24,anomaly_count=90)
    samples={}
    for index,row in enumerate(rows):
        if not int(row['gt_label']):continue
        sid=row['sample_id'];score=np.load(row['raw_map_path'],allow_pickle=False);labels,sizes=exact.components(exact.get_mask(row,Image,np),cv2,np)
        fg=labels>0;values=score[fg];rw=1/((len(sizes)-1)*sizes[labels[fg]].astype(np.float64))
        points=exact.pro_at_fprs(bg,values,rw,[.05,.30],np)
        localbg=np.sort(score[~fg]);local,_=exact.integrate(localbg,values,rw,np)
        samples[sid]=dict(sample_id=sid,branch=tag,size_group=membership[sid]['size_group'],defect_ratio=float(membership[sid]['defect_ratio']),PRO005=float(points[0]),PRO030=float(points[1]),per_image_AUPRO005=local[.05],per_image_AUPRO030=local[.30],pixel_auroc=float(diagnostic.percentiles(localbg,values)[2].mean()/100),percentile_p95=float(np.percentile(diagnostic.percentiles(bg,values)[2],95)))
        if (index+1)%15==0:print(f'{tag}: paired metrics {index+1}/114',flush=True)
    if any(not 0<=result[k]<=1 for k in ('image_auroc','pixel_auroc','aupro005','aupro030')):raise ValueError('Metric outside[0,1]')
    return result,samples,curve,meta

def old_st_curve(rows,frozen):
    """Read trusted frozen cache, verify its integral before using PRO points."""
    meta=frozen['exact_metadata']['ST_NORMALIZED'] if 'exact_metadata' in frozen else None
    if meta is None:
        # Derive exact counts from official masks, without altering frozen cache.
        import cv2
        counts=[exact.components(exact.get_mask(r,Image,np),cv2,np)[1] for r in rows]
        nb=sum(int(a[0]) for a in counts);n=sum(int(a[1:].sum()) for a in counts)
    else:nb=meta['background_pixels'];n=meta['positive_pixels']
    folder=D10/'temp/st_normalized'
    specs=[('background_scores.dat',np.float32,nb),('positive_scores.dat',np.float32,n),('positive_weights.dat',np.float64,n)]
    arrays=[]
    for name,dtype,count in specs:
        p=folder/name
        if p.stat().st_size!=count*np.dtype(dtype).itemsize:raise ValueError('Frozen exact cache size mismatch')
        arrays.append(np.memmap(p,mode='r',dtype=dtype,shape=(count,)))
    bg,pos,w=arrays
    for i in range(0,len(bg)-1,1000000):
        stop=min(len(bg)-1,i+1000000)
        if np.any(bg[i:stop]>bg[i+1:stop+1]):raise ValueError('Frozen cache not sorted')
    areas,total=exact.integrate(bg,pos,w,np)
    if abs(total-1)>1e-6 or any(abs(areas[a]-frozen['overall']['ST_NORMALIZED'][key])>1e-7 for a,key in ((.05,'aupro005'),(.30,'aupro030'))):raise ValueError('Frozen exact reference cache fails metric identity')
    return exact.pro_at_fprs(bg,pos,w,diagnostic.FPRS,np).tolist()
