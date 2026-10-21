import csv,gc,hashlib,json,math,os,sys,time,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[1]
P=R/'experiments/phase24a_p1_r1_patchcore_fulltrain_memory'
O=R/'experiments/phase24a_p2_patchcore_geometry_formal'
D=R/'datasets/MVTec_AD_2/sheet_metal'
H={'square':'00092b9231034e777823733e0616efaf7612c0736496550e88157beeab39d332','rect':'9feadc0e921a8ee16de074ad856ed466b1f7e95484d1c23a2d69cf7dfdf86f33'}
W='03b71d65fb2c73bb0de079a1781009f27a782ec481d2f64ab3bde9b1cdec3000'
accessed=[];blocked=[]
def audit(event,args):
    if event=='socket.connect': raise PermissionError('offline')
    if event not in ('open','os.listdir','os.scandir') or not args or not isinstance(args[0],(str,bytes)): return
    p=Path(os.fsdecode(args[0])).absolute()
    if p.is_relative_to(R/'datasets'):
        if not (p.is_relative_to(D/'test_public') or p.is_relative_to(D/'validation/good')):
            blocked.append(str(p));raise PermissionError(str(p))
        if event=='open': accessed.append(str(p))
sys.addaudithook(audit)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(n,x):
    p=O/n;p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(str(p))
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def gates():
    c=read(P/'completion_verification.json')
    t=(P/'final_console_report.txt').read_text(encoding='utf8')
    ok=(c['status']=='PASS_FOR_PATCHCORE_GEOMETRY_FORMAL' and c['status'] in t
        and c['FULL_RUN_8GB_FEASIBLE'] is True and c['PHASE24A_P2_AUTHORIZED'] is True
        and c['EQUAL_RAW_EMBEDDING_COUNT'] is True and c['EQUAL_MEMORY_BANK_COUNT'] is True
        and read(P/'01_previous_gate.json')['checks']['equal_pixels'] is True
        and c['test_public_files_accessed']==c['test_mask_files_accessed']==c['protocol_violation_count']==0)
    if not ok:raise RuntimeError('FAIL_PREVIOUS_PHASE_GATE')
    cfg=c['config']
    assert cfg==dict(backbone='wide_resnet50_2.racm_in1k',layers=['layer2','layer3'],coreset_ratio=.1,num_neighbors=9,precision='FP32',weight_sha256=W,anomalib='2.6.1',square=[512,512],rect=[256,1024],seed=42,pretrained=True)
    banks={}
    for tag in H:
        f=P/(tag+'_full_memory_bank.pt');actual=sha(f)
        banks[tag]=dict(path=str(f),sha256=actual,expected_sha256=H[tag],sha_match=actual==H[tag],size_bytes=f.stat().st_size)
    if not all(x['sha_match'] for x in banks.values()):raise RuntimeError('FAIL_MEMORY_BANK_SHA_GATE')
    return c,banks
def public():
    rows=read(R/'experiments/phase20b_p2_patchcore_formal/prediction_manifest.json')
    files=set((D/'test_public/good').glob('*.png'))|set((D/'test_public/bad').glob('*.png'))
    masks=set((D/'test_public/ground_truth/bad').glob('*.png'))
    assert len(rows)==len(files)==114 and len(masks)==90
    assert {Path(x['image_path']) for x in rows}==files
    assert {Path(x['gt_mask_path']) for x in rows if x['gt_label']}==masks
    assert sum(x['gt_label'] for x in rows)==90
    assert len({x['sample_id'] for x in rows})==114
    return rows
def model_for(tag):
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['HF_HUB_CACHE']=str(R/'experiments/phase20b_p1_r3_patchcore_smoke_resume/weights/hub')
    import importlib.metadata as md,inspect,torch
    from anomalib.models import Patchcore
    from safetensors.torch import load_file
    assert torch.cuda.is_available() and md.version('anomalib')=='2.6.1'
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['HF_HUB_CACHE']=str(R/'experiments/phase20b_p1_r3_patchcore_smoke_resume/weights/hub')
    wp=Path(read(R/'experiments/phase20b_p2_patchcore_formal/completion_verification.json')['pretrained_weight_cache_or_path'])
    assert sha(wp)==W
    hw=(512,512) if tag=='square' else (256,1024)
    m=Patchcore(backbone='wide_resnet50_2.racm_in1k',layers=['layer2','layer3'],pre_trained=False,coreset_sampling_ratio=.1,num_neighbors=9,precision='float32',pre_processor=Patchcore.configure_pre_processor(image_size=hw,center_crop_size=None),post_processor=False,evaluator=False,visualizer=False)
    fe=m.model.feature_extractor.feature_extractor;weights=load_file(str(wp),device='cpu');state=fe.state_dict()
    assert len(state)==258
    mapped={k:weights[k if k in weights else k.replace('_','.')] for k in state}
    fe.load_state_dict(mapped,strict=True)
    assert all(torch.equal(v,mapped[k]) for k,v in fe.state_dict().items())
    prior=read(P/('02_square512_full_build.json' if tag=='square' else '03_rect_full_build.json'))
    assert repr(m.pre_processor.transform)==prior['preprocessor']
    assert hashlib.sha256(inspect.getsource(m.model.compute_anomaly_score).encode()).hexdigest()==prior['score_source_sha256']
    bank=torch.load(P/(tag+'_full_memory_bank.pt'),map_location='cpu',weights_only=True)
    assert bank.dtype==torch.float32 and list(bank.shape)==prior['memory_bank_shape']==[56115,1536] and torch.isfinite(bank).all()
    m.model.memory_bank=bank.cuda();m=m.cuda().eval()
    assert not m.model.embedding_store
    return m,hw
def predict(tag,source):
    import numpy as np,torch
    import torch.nn.functional as F
    from anomalib.data.utils import read_image
    m,hw=model_for(tag);folder=O/('04_square512_predictions' if tag=='square' else '05_rect_predictions')
    folder.mkdir()
    val=sorted((D/'validation/good').glob('*.png'));assert len(val)==19
    def load(p):
        x=read_image(p,as_tensor=True);assert tuple(x.shape[-2:])==(1056,4224)
        return m.pre_processor.transform(x).unsqueeze(0).cuda()
    def forward(x):
        y=m.model(x);raw=y.anomaly_map.detach().float()
        assert tuple(raw.shape)==(1,1,*hw)
        return y,raw
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    with torch.inference_mode():
        for p in val[:10]:forward(load(p))
        torch.cuda.synchronize()
        rows=[];mts=[];ets=[]
        for round in range(3):
            for i,s in enumerate(source):
                torch.cuda.synchronize();start=time.perf_counter()
                x=load(Path(s['image_path']))
                torch.cuda.synchronize();ms=time.perf_counter()
                y,raw=forward(x)
                torch.cuda.synchronize();mt=(time.perf_counter()-ms)*1000
                native=F.interpolate(raw,size=(1056,4224),mode='bilinear',align_corners=False)
                arr=native[0,0].cpu().numpy().copy();score=float(y.pred_score.item())
                torch.cuda.synchronize();et=(time.perf_counter()-start)*1000
                assert math.isfinite(score) and np.isfinite(arr).all()
                mts.append(mt);ets.append(et)
                if round==0:
                    rawarr=raw[0,0].cpu().numpy().copy()
                    assert np.isfinite(rawarr).all()
                    rp=folder/(s['sample_id']+'_model.npy');npth=folder/(s['sample_id']+'_native.npy')
                    assert not rp.exists() and not npth.exists()
                    np.save(rp,rawarr);np.save(npth,arr)
                    rows.append(dict(sample_id=s['sample_id'],image_path=s['image_path'],gt_label=s['gt_label'],gt_mask_path=s['gt_mask_path'],official_image_score=score,model_map_shape=list(rawarr.shape),native_shape=[1056,4224],raw_model_map_path=str(rp),raw_map_path=str(npth),model_map_sha256=sha(rp),native_map_sha256=sha(npth)))
                if (i+1)%20==0:print(tag,'ROUND',round+1,i+1,flush=True)
    assert len(rows)==114 and len(mts)==len(ets)==342
    rt=dict(warmup_validation_count=10,rounds=3,test_count_per_round=114,model_only_latency_mean_ms=float(np.mean(mts)),model_only_latency_std_ms=float(np.std(mts)),model_only_fps=float(1000/np.mean(mts)),e2e_latency_mean_ms=float(np.mean(ets)),e2e_latency_std_ms=float(np.std(ets)),e2e_fps=float(1000/np.mean(ets)),peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_cuda_reserved_mib=torch.cuda.max_memory_reserved()/2**20,memory_bank_mib=m.model.memory_bank.numel()*m.model.memory_bank.element_size()/2**20,model_only_samples_ms=mts,e2e_samples_ms=ets)
    save(('09_square512_runtime.json' if tag=='square' else '10_rect_runtime.json'),rt)
    save(('04_square512_predictions/manifest.json' if tag=='square' else '05_rect_predictions/manifest.json'),rows)
    del m;gc.collect();torch.cuda.empty_cache()
    return rows,rt
def metric(tag,rows):
    import cv2,numpy as np,torch
    from PIL import Image
    from sklearn.metrics import roc_auc_score
    sys.path.insert(0,str(R))
    from src.evaluation import exact_aupro as exact
    assert sha(R/'src/evaluation/exact_aupro.py')=='3e6ebc80d400419cf3aaf00950e954ea165df2419c6ea895f4296013c67e11a1'
    folder=O/('04_square512_predictions' if tag=='square' else '05_rect_predictions')
    scores=np.lib.format.open_memmap(folder/'native_scores.npy',mode='w+',dtype=np.float32,shape=(114,1056,4224))
    targets=np.lib.format.open_memmap(folder/'native_targets.npy',mode='w+',dtype=np.uint8,shape=(114,1056,4224))
    for i,row in enumerate(rows):
        arr=np.load(row['raw_map_path'],allow_pickle=False)
        assert arr.shape==(1056,4224) and arr.dtype==np.float32 and np.isfinite(arr).all()
        assert sha(row['raw_map_path'])==row['native_map_sha256']
        mask=exact.get_mask(row,Image,np);assert mask.shape==(1056,4224)
        if row['gt_label']:assert mask.any()
        scores[i]=arr;targets[i]=mask
    scores.flush();targets.flush()
    im=float(roc_auc_score([r['gt_label'] for r in rows],[r['official_image_score'] for r in rows]))
    print(tag,'PIXEL_AUROC_START',flush=True)
    px=float(roc_auc_score(targets.reshape(-1),scores.reshape(-1)))
    del scores,targets;gc.collect()
    er={};exact.native(rows,er,torch,cv2,np,Image,temp_dir=folder/'exact_temp')
    assert er['connected_regions']==450
    result=dict(image_auroc=im,pixel_auroc=px,exact_aupro005=er['aupro_0.05'],exact_aupro030=er['aupro_0.30'],gt_region_count=450,background_pixels=er['background_pixels'],exact_implementation_sha256=sha(R/'src/evaluation/exact_aupro.py'),image_score_definition='OFFICIAL_PATCHCORE_WEIGHTED_NN_SCORE',native_restore=dict(interpolation='bilinear',align_corners=False,smoothing=False,morphology=False))
    save('06_square512_metrics.json' if tag=='square' else '07_rect_metrics.json',result)
    return result
def main():
    if O.exists():
        assert {p.name for p in O.iterdir()}=={'01_preregistration.json','02_bank_verification.json','03_data_audit.json'}
    c,banks=gates();source=public()
    O.mkdir(exist_ok=True)
    if not (O/'01_preregistration.json').exists():save('01_preregistration.json',dict(phase='Phase24A-P2',square_geometry=[512,512],rect_geometry=[256,1024],input_pixels=262144,backbone=c['config']['backbone'],layers=c['config']['layers'],coreset_ratio=.1,num_neighbors=9,precision='FP32',image_score_definition='OFFICIAL_PATCHCORE_WEIGHTED_NN_SCORE',native_restore=[1056,4224],metric_source_sha256=sha(R/'src/evaluation/exact_aupro.py'),core_comparison='Rect minus Square',success_threshold=None,bootstrap=False,low_budget_reference=[71.759259,83.579292,6.609418,21.420678]))
    if not (O/'02_bank_verification.json').exists():save('02_bank_verification.json',banks)
    if not (O/'03_data_audit.json').exists():save('03_data_audit.json',dict(dataset='MVTec AD 2',class_name='sheet_metal',test_public_image_count=114,test_public_normal_count=24,test_public_anomaly_count=90,gt_mask_image_count=90,test_private_files_accessed=0,test_private_mixed_files_accessed=0))
    sr,srt=predict('square',source);sm=metric('square',sr)
    rr,rrt=predict('rect',source);rm=metric('rect',rr)
    assert [r['sample_id'] for r in sr]==[r['sample_id'] for r in rr]==[r['sample_id'] for r in source]
    delta={k+'_pp':100*(rm[k]-sm[k]) for k in ['image_auroc','pixel_auroc','exact_aupro005','exact_aupro030']}
    save('08_geometry_delta.json',delta)
    save('11_runtime_comparison.json',dict(square=srt,rect=rrt))
    integ=dict(square_prediction_count=114,rect_prediction_count=114,missing_count=0,duplicate_count=0,nan_image_scores=0,inf_image_scores=0,nan_anomaly_pixels=0,inf_anomaly_pixels=0,prediction_alignment_pass=True)
    save('12_prediction_integrity.json',integ)
    assert not blocked and not any('test_private' in p for p in accessed)
    save('13_protocol_audit.json',dict(accessed=accessed,blocked=blocked,test_private_files_accessed=0,test_private_mixed_files_accessed=0,protocol_violation_count=0,image_score_pipeline_identical=True,memory_bank_rebuilt=False,coreset_rerun=False,fastflow_started=False,bootstrap_run=False))
    v=dict(phase='Phase24A-P2',status='PASS_PATCHCORE_GEOMETRY_FORMAL_COMPLETE',square_bank_sha_match=True,rect_bank_sha_match=True,square_geometry=[512,512],rect_geometry=[256,1024],square_pixels=262144,rect_pixels=262144,equal_input_pixel_budget=True,backbone=c['config']['backbone'],pretrained_tag='racm_in1k',pretrained_weight_sha256=W,layers=['layer2','layer3'],coreset_ratio=.1,num_neighbors=9,square_bank_shape=[56115,1536],rect_bank_shape=[56115,1536],test_public_count=114,test_normal_count=24,test_anomaly_count=90,gt_region_count=450,square_prediction_count=114,rect_prediction_count=114,prediction_alignment_pass=True,test_private_files_accessed=0,test_private_mixed_files_accessed=0,protocol_violation_count=0,square_image_score_definition='OFFICIAL_PATCHCORE_WEIGHTED_NN_SCORE',rect_image_score_definition='OFFICIAL_PATCHCORE_WEIGHTED_NN_SCORE',image_score_pipeline_identical=True)
    for tag,m,rt in [('square',sm,srt),('rect',rm,rrt)]:
        for k in ['image_auroc','pixel_auroc','exact_aupro005','exact_aupro030']:v[tag+'_'+k]=m[k]
        for stem,key in [('model_latency_mean_ms','model_only_latency_mean_ms'),('model_latency_std_ms','model_only_latency_std_ms'),('model_fps','model_only_fps'),('e2e_latency_mean_ms','e2e_latency_mean_ms'),('e2e_latency_std_ms','e2e_latency_std_ms'),('e2e_fps','e2e_fps'),('peak_allocated_mib','peak_cuda_allocated_mib')]:v[tag+'_'+stem]=rt[key]
    for k,x in delta.items():v['delta_'+k]=x
    save('completion_verification.json',v)
    lines=['PHASE24A-P2 PATCHCORE GEOMETRY FORMAL','STATUS = '+v['status']]
    for title,m,rt in [('Square512',sm,srt),('Rect256x1024',rm,rrt)]:
        lines.append(title)
        for k in ['image_auroc','pixel_auroc','exact_aupro005','exact_aupro030']:lines.append(k+' = '+format(100*m[k],'.6f')+'%')
        lines.extend(['model latency = %.6f ± %.6f ms; FPS = %.6f'%(rt['model_only_latency_mean_ms'],rt['model_only_latency_std_ms'],rt['model_only_fps']),'E2E latency = %.6f ± %.6f ms; FPS = %.6f'%(rt['e2e_latency_mean_ms'],rt['e2e_latency_std_ms'],rt['e2e_fps']),'peak CUDA allocated = %.6f MiB'%rt['peak_cuda_allocated_mib']])
    lines+=['Rect - Square']+[k+' = %+.6f pp'%x for k,x in delta.items()]
    lines+=['predictions = 114/114 each; NaN/Inf = 0; bank SHA gates = True','private/private_mixed accesses = 0/0; protocol violations = 0','final_console_report.txt = '+str(O/'final_console_report.txt'),'completion_verification.json = '+str(O/'completion_verification.json')]
    (O/'final_console_report.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('\n'.join(lines),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:traceback.print_exc();raise
