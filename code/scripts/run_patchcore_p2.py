"""Authorized Phase20B-P1 only. No test dataset loader, optimizer, metrics or full fit."""
import os,sys,json,hashlib,time,math,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'datasets/MVTec_AD_2/sheet_metal'
OUT=ROOT/'experiments/phase20b_p2_patchcore_formal'
TEST_UNLOCKED=False
def assert_allowed_data_path(path):
    p=Path(path).resolve()
    try: rel=p.relative_to(DATA.resolve())
    except ValueError: raise PermissionError('Dataset outside sheet_metal normal-only scope')
    if not rel.parts or rel.parts[0] not in (('train','validation','test_public') if TEST_UNLOCKED else ('train','validation')):
        raise PermissionError('Forbidden data role: '+str(p))
    return p
def select_subset(paths,count):
    paths=sorted(paths,key=lambda p:str(p))
    if len(paths)<count: raise ValueError('Insufficient sources')
    return paths[:count]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def main():
    global TEST_UNLOCKED
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'completion_verification.json').exists(): raise RuntimeError('Existing completion: do not repeat smoke automatically')
    accessed=[]; blocked=[]
    def audit(event,args):
        if event == 'socket.connect': raise PermissionError('R3 is strictly offline')
        if event not in ('open','os.listdir','os.scandir') or not args or not isinstance(args[0],(str,bytes)): return
        p=Path(os.fsdecode(args[0])).absolute()
        if p.is_relative_to(ROOT/'datasets'):
            try: assert_allowed_data_path(p)
            except PermissionError:
                blocked.append(str(p)); raise
            if event=='open': accessed.append(str(p))
    sys.addaudithook(audit)
    stage='Phase20A gate'
    c=dict(phase='Phase20B-P2',status='HOLD_PATCHCORE_PROTOCOL',dataset='MVTec AD 2',class_name='sheet_metal',
      backbone='wide_resnet50_2.racm_in1k',layers=['layer2','layer3'],pre_trained=True,coreset_sampling_ratio=0.1,num_neighbors=9,
      precision='float32',input_height=256,input_width=256,center_crop=None,seed=42,
      optimizer_created=False,backward_count=0,optimizer_step_count=0,val_forward_count=0,
      test_public_files_accessed=0,test_mask_files_accessed=0,phase20b_p2_patchcore_formal_authorized=False,protocol_violation_count=0)
    try:
        r3=ROOT/'experiments/phase20b_p1_r3_patchcore_smoke_resume'
        prior=json.loads((r3/'completion_verification.json').read_text(encoding='utf8'))
        (r3/'final_console_report.txt').read_text(encoding='utf8')
        assert prior['status']=='PASS_FOR_PATCHCORE_FORMAL' and prior['phase20b_p2_patchcore_formal_authorized'], 'FAIL_PREVIOUS_GATE'
        assert prior['test_public_files_accessed']==prior['test_mask_files_accessed']==0
        pa=ROOT/'experiments/phase20a_external_baseline_audit'
        gate=json.loads((pa/'completion_verification.json').read_text(encoding='utf8'))
        text=(pa/'final_console_report.txt').read_text(encoding='utf8')
        assert gate['status']=='PASS_FOR_EXTERNAL_BASELINE_SMOKE' and gate['patchcore_available']
        assert gate['anomalib_version']=='2.6.1' and gate['recommended_input_protocol']=='256x256'
        save('phase20a_gate.json',dict(status='PASS',report_sha256=sha(pa/'final_console_report.txt'),completion_sha256=sha(pa/'completion_verification.json')))
        r2=ROOT/'experiments/phase20b_p1_r2_patchcore_direct_weight'
        original=ROOT/'experiments/phase20b_p1_patchcore_smoke'
        old=json.loads((original/'completion_verification.json').read_text())
        recovered=json.loads((r2/'completion_verification.json').read_text())
        (original/'final_console_report.txt').read_text(encoding='utf8')
        (r2/'final_console_report.txt').read_text(encoding='utf8')
        assert old['status']=='HOLD_PATCHCORE_PRETRAINED_WEIGHT'
        assert recovered['status']=='PASS_FOR_PATCHCORE_SMOKE_RESUME' and recovered['patchcore_smoke_resume_authorized']
        wp=Path(recovered['weight_path'])
        weight_sha=sha(wp)
        assert weight_sha=='03b71d65fb2c73bb0de079a1781009f27a782ec481d2f64ab3bde9b1cdec3000', 'FAIL_WEIGHT_SHA_GATE'
        previous_hashes={str(d/n):sha(d/n) for d in [original,r2] for n in ['final_console_report.txt','completion_verification.json']}
        save('01_preflight.json',dict(status='PASS',original_status=old['status'],R2_status=recovered['status'],source_sha256=previous_hashes))
        # Byte-identical local cache link, not a download or weight replacement.
        cache=r3/'weights/hub'
        repo=cache/'models--timm--wide_resnet50_2.racm_in1k'
        commit='30f73aceaaa1911830a9795b83ab1908dba18719'
        snapshot=repo/'snapshots'/commit/'model.safetensors'
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        if not snapshot.exists(): os.link(wp,snapshot)
        assert sha(snapshot)==weight_sha
        (repo/'refs').mkdir(exist_ok=True)
        assert (repo/'refs/main').read_text()==commit
        os.environ['HF_HUB_CACHE']=str(cache)
        os.environ['HF_HUB_OFFLINE']='1'
        fallback=None
        import numpy as np,torch,importlib.metadata as md
        from lightning import seed_everything
        from anomalib.models import Patchcore
        from anomalib.data.utils import read_image
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        import inspect
        assert md.version('anomalib')=='2.6.1'
        c.update(anomalib_version=md.version('anomalib'),patchcore_source=inspect.getfile(Patchcore))
        source_paths=[Path(inspect.getfile(Patchcore)),ROOT/'src/evaluation/exact_aupro.py',pa/'completion_verification.json']
        hashes={str(p):sha(p) for p in source_paths}
        save('03_patchcore_config.json',dict(c,post_processor=False,evaluator=False,visualizer=False,
          preprocessor='official configure_pre_processor(image_size=(256,256), center_crop_size=None)',source_sha256=hashes))
        stage='pretrained weights'
        initialization_start=time.perf_counter()
        seed_everything(42,workers=True)
        model=Patchcore(backbone='wide_resnet50_2.racm_in1k',layers=['layer2','layer3'],pre_trained=True,
          coreset_sampling_ratio=0.1,num_neighbors=9,precision='float32',
          pre_processor=Patchcore.configure_pre_processor(image_size=(256,256),center_crop_size=None),
          post_processor=False,evaluator=False,visualizer=False)
        weights=load_file(str(wp),device='cpu')
        extracted=model.model.feature_extractor.feature_extractor.state_dict()
        matches=[]
        for key,tensor in extracted.items():
            original=key if key in weights else key.replace('_','.')
            assert original in weights and torch.equal(tensor.cpu(),weights[original]), 'Pretrained tensor mismatch '+key
            matches.append(key)
        del weights,extracted
        wa=dict(pretrained_weight_available=True,pretrained_weight_source=('official timm GitHub release: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/wide_resnet50_racm-8234f177.pth' if wp==fallback else 'official timm Hugging Face repository'),
          pretrained_weight_identifier='timm/wide_resnet50_2.racm_in1k',pretrained_weight_cache_or_path=str(wp),
          pretrained_weight_sha256=weight_sha,repository_revision=commit,
          loaded_backbone_tensor_count=len(matches),all_loaded_backbone_tensors_exact_match=True,random_backbone_fallback=False)
        save('02_weight_verification.json',wa); c.update(wa); c['model_initialization_time_sec']=time.perf_counter()-initialization_start
        stage='postprocessor'
        assert model.post_processor is None and model.evaluator is None and model.visualizer is None
        assert model.configure_optimizers() is None
        c['postprocessor_test_independent']=True
        assert 'CenterCrop' not in repr(model.pre_processor.transform)
        stage='manifest'
        trains=sorted((DATA/'train/good').glob('*.png')); vals=sorted((DATA/'validation/good').glob('*.png'))
        assert len(trains)==137 and len(vals)==19
        train=select_subset(trains,137); val=[]
        manifest={role:[dict(path=str(assert_allowed_data_path(p)),sha256=sha(p)) for p in ps] for role,ps in [('train',train),('validation',val)]}
        save('04_train_manifest.json',dict(manifest,selection='all137 normal training sources; no calibration fitting',train_total=137,validation_total=19))
        c.update(train_count=137,validation_count=19)
        # Fail-closed counters prevent accidental optimizer or backward usage by any helper.
        def forbidden_backward(*a,**kw):
            c['backward_count']+=1; raise RuntimeError('Backward forbidden')
        def forbidden_optimizer(*a,**kw):
            c['optimizer_created']=True; raise RuntimeError('Optimizer forbidden')
        torch.autograd.backward=forbidden_backward
        torch.optim.Optimizer.__init__=forbidden_optimizer
        gpu_init=time.perf_counter(); model=model.cuda(); torch.cuda.synchronize()
        c['model_initialization_time_sec']+=time.perf_counter()-gpu_init
        def sync(): torch.cuda.synchronize()
        def peak():
            import ctypes
            class Memory(ctypes.Structure):
                _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(k,ctypes.c_size_t) for k in ['peak','working','a','b','d','e','f','g']]
            info=Memory();info.cb=ctypes.sizeof(info)
            kernel=ctypes.WinDLL('kernel32');kernel.GetCurrentProcess.restype=ctypes.c_void_p
            psapi=ctypes.WinDLL('psapi')
            psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(Memory),ctypes.c_ulong]
            ok=psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(info),info.cb)
            return dict(host_rss_mib=info.working/2**20 if ok else None,host_process_peak_mib=info.peak/2**20 if ok else None,allocated_mib=torch.cuda.max_memory_allocated()/2**20,reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        def tensor_state_hash():
            h=hashlib.sha256()
            for k,v in model.model.feature_extractor.state_dict().items(): h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
            return h.hexdigest()
        backbone_before=tensor_state_hash()
        def load(p):
            assert_allowed_data_path(p)
            image=read_image(p,as_tensor=True)
            return model.pre_processor.transform(image).unsqueeze(0).cuda()
        stage='feature extraction'; torch.cuda.reset_peak_memory_stats(); sync(); fit_start=time.perf_counter()
        model.model.train(); model.model.feature_extractor.eval()
        with torch.no_grad():
            for i,p in enumerate(train):
                x=load(p); assert tuple(x.shape)==(1,3,256,256)
                emb=model.model(x); assert torch.isfinite(emb).all()
                print('TRAIN_FEATURE',i+1,tuple(emb.shape),flush=True)
        sync(); ft=time.perf_counter()-fit_start; feature_peak=peak()
        raw_count=sum(e.shape[0] for e in model.model.embedding_store); dim=emb.shape[1]
        c.update(raw_embedding_shape=[raw_count,dim],raw_embedding_count=raw_count)
        c['embedding_cache_device']=str(model.model.embedding_store[0].device)
        del x,emb
        stage='coreset'; torch.cuda.reset_peak_memory_stats(); sync(); start=time.perf_counter()
        with torch.no_grad(): model.fit()
        sync(); ct=time.perf_counter()-start; coreset_peak=peak(); totalfit=time.perf_counter()-fit_start
        bank=model.model.memory_bank
        assert bank.shape[0]==int(raw_count*0.1) and bank.numel()>0 and torch.isfinite(bank).all()
        c.update(memory_bank_shape=list(bank.shape),memory_bank_count=bank.shape[0],actual_coreset_ratio=bank.shape[0]/raw_count,memory_bank_device=str(bank.device))
        save('05_memory_bank_summary.json',dict(raw_embedding_shape=[raw_count,dim],memory_bank_shape=list(bank.shape),
          raw_embedding_count=raw_count,memory_bank_count=bank.shape[0],finite=True,coreset_ratio=0.1,actual_coreset_ratio=bank.shape[0]/raw_count,embedding_cache_device=c['embedding_cache_device'],memory_bank_device=str(bank.device),
          optimizer_created=False,backward_count=0,optimizer_step_count=0))
        c.update(model_init_sec=c['model_initialization_time_sec'],feature_extraction_sec=ft,coreset_build_sec=ct,total_build_sec=totalfit,
          fit_peak_allocated_mib=max(feature_peak['allocated_mib'],coreset_peak['allocated_mib']),
          fit_peak_reserved_mib=max(feature_peak['reserved_mib'],coreset_peak['reserved_mib']))
        save('06_fit_runtime_memory.json',dict(c,feature_peak=feature_peak,coreset_peak=coreset_peak))
        checkpoint=OUT/'formal_memory_bank.pt'
        torch.save(dict(state_dict=model.state_dict(),config=json.loads((OUT/'03_patchcore_config.json').read_text())),checkpoint)
        loaded=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model.load_state_dict(loaded['state_dict'],strict=True)
        assert all(torch.equal(v.cpu(),loaded['state_dict'][k]) for k,v in model.state_dict().items())
        del loaded
        c.update(checkpoint_path=str(checkpoint),checkpoint_sha256=sha(checkpoint),strict_reload='PASS')
        assert tensor_state_hash()==backbone_before
        TEST_UNLOCKED=True
        stage='formal inference and metrics'
        from patchcore_p2_evaluation import finish
        finish(model,c,OUT,ROOT,DATA,load,sync,peak,sha,save)
        assert all(sha(p)==v for p,v in hashes.items()) and not blocked
        assert all(sha(p)==v for p,v in previous_hashes.items()) and sha(wp)==weight_sha
        c['status']='PASS'
    except Exception as e:
        c.update(error=str(e),failed_stage=stage,status='HOLD_PATCHCORE_PROTOCOL')
        if 'out of memory' in str(e).lower():
            c['status']='HOLD_PATCHCORE_FULL_OOM'
        for gate in ['FAIL_PREVIOUS_GATE','FAIL_WEIGHT_SHA_GATE']:
            if gate in str(e): c['status']=gate
        if 'torch' in locals() and torch.cuda.is_available():
            c.update(failure_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,failure_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        traceback.print_exc()
    finally:
        c.update(test_dependent_normalization=False,test_dependent_threshold=False,test_dependent_calibration=False,
          test_public_files_accessed=sum('test_public' in p and 'ground_truth' not in p for p in accessed),
          test_mask_files_accessed=sum('ground_truth' in p for p in accessed),protocol_violation_count=len(blocked))
        names=['01_preflight','02_weight_verification','03_patchcore_config','04_train_manifest','05_memory_bank_summary','06_fit_runtime_memory','07_test_inference_summary','08_image_score_protocol','09_native_restore_summary','10_evaluation_report','11_aupro_exact_report','12_runtime_memory','13_integrity_audit','14_comparison_with_existing','15_external_comparison_table']
        for name in names:
            if not (OUT/(name+'.json')).exists(): save(name+'.json',dict(status='NOT_EXECUTED',reason=c.get('error')))
        save('16_protocol_audit.json',dict(dataset_access_events=accessed,blocked=blocked,public_unlocked_only_after_full_bank=TEST_UNLOCKED,
          optimizer_created=c['optimizer_created'],backward_count=c['backward_count'],optimizer_step_count=c['optimizer_step_count'],
          test_dependent_normalization=False,test_dependent_threshold=False,test_dependent_calibration=False,protocol_violation_count=len(blocked)))
        save('completion_verification.json',c)
        (OUT/'final_console_report.txt').write_text('PHASE20B-P2 PATCHCORE FORMAL EXTERNAL BASELINE\n'+json.dumps(c,indent=2,ensure_ascii=False)+'\nMethod-level comparison; no configuration search; FastFlow not started.\n',encoding='utf8')
        print(json.dumps(c,indent=2,ensure_ascii=False),flush=True)
    return 0 if c['status']=='PASS' else 1
if __name__=='__main__': sys.exit(main())
