"""Authorized Phase20B-P1 only. No test dataset loader, optimizer, metrics or full fit."""
import os,sys,json,hashlib,time,math,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'datasets/MVTec_AD_2/sheet_metal'
OUT=ROOT/'experiments/phase20b_p1_r3_patchcore_smoke_resume'
def assert_allowed_data_path(path):
    p=Path(path).resolve()
    try: rel=p.relative_to(DATA.resolve())
    except ValueError: raise PermissionError('Dataset outside sheet_metal normal-only scope')
    if not rel.parts or rel.parts[0] not in ('train','validation') or any(x in ('ground_truth','bad') for x in rel.parts):
        raise PermissionError('Forbidden data role: '+str(p))
    return p
def select_subset(paths,count):
    paths=sorted(paths,key=lambda p:str(p))
    if len(paths)<count: raise ValueError('Insufficient sources')
    return paths[:count]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def main():
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
    c=dict(phase='Phase20B-P1-R3',status='HOLD_PATCHCORE_PROTOCOL',dataset='MVTec AD 2',class_name='sheet_metal',
      backbone='wide_resnet50_2.racm_in1k',layers=['layer2','layer3'],pre_trained=True,coreset_sampling_ratio=0.1,num_neighbors=9,
      precision='float32',input_height=256,input_width=256,center_crop=None,seed=42,
      optimizer_created=False,backward_count=0,optimizer_step_count=0,val_forward_count=0,
      test_public_files_accessed=0,test_mask_files_accessed=0,phase20b_p2_patchcore_formal_authorized=False,protocol_violation_count=0)
    try:
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
        save('01_previous_gates.json',dict(status='PASS',original_status=old['status'],R2_status=recovered['status'],source_sha256=previous_hashes))
        # Byte-identical local cache link, not a download or weight replacement.
        cache=OUT/'weights/hub'
        repo=cache/'models--timm--wide_resnet50_2.racm_in1k'
        commit='30f73aceaaa1911830a9795b83ab1908dba18719'
        snapshot=repo/'snapshots'/commit/'model.safetensors'
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        if not snapshot.exists(): os.link(wp,snapshot)
        assert sha(snapshot)==weight_sha
        (repo/'refs').mkdir(exist_ok=True)
        (repo/'refs/main').write_text(commit)
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
        save('04_patchcore_config.json',dict(c,post_processor=False,evaluator=False,visualizer=False,
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
        save('02_weight_reverification.json',wa); c.update(wa); c['model_initialization_time_sec']=time.perf_counter()-initialization_start
        stage='postprocessor'
        assert model.post_processor is None and model.evaluator is None and model.visualizer is None
        assert model.configure_optimizers() is None
        c['postprocessor_test_independent']=True
        assert 'CenterCrop' not in repr(model.pre_processor.transform)
        stage='manifest'
        trains=sorted((DATA/'train/good').glob('*.png')); vals=sorted((DATA/'validation/good').glob('*.png'))
        assert len(trains)==137 and len(vals)==19
        train=select_subset(trains,16); val=select_subset(vals,4)
        manifest={role:[dict(path=str(assert_allowed_data_path(p)),sha256=sha(p)) for p in ps] for role,ps in [('train',train),('validation',val)]}
        save('03_smoke_manifest.json',dict(manifest,selection='lexicographic first16 train and first4 validation',train_total=137,validation_total=19))
        c.update(train_smoke_count=16,validation_smoke_count=4)
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
        save('05_memory_bank_audit.json',dict(raw_embedding_shape=[raw_count,dim],memory_bank_shape=list(bank.shape),
          raw_embedding_count=raw_count,memory_bank_count=bank.shape[0],finite=True,coreset_ratio=0.1,actual_coreset_ratio=bank.shape[0]/raw_count,embedding_cache_device=c['embedding_cache_device'],memory_bank_device=str(bank.device),
          optimizer_created=False,backward_count=0,optimizer_step_count=0))
        stage='validation'; torch.cuda.reset_peak_memory_stats(); model.eval(); rows=[]
        (OUT/'validation_maps').mkdir(exist_ok=True)
        with torch.no_grad():
            for p in val:
                sync(); e0=time.perf_counter(); x=load(p); sync(); m0=time.perf_counter()
                output=model.model(x); sync(); mt=time.perf_counter()-m0; c['val_forward_count']+=1
                raw=output.anomaly_map.detach().float(); score=float(output.pred_score.item())
                assert tuple(raw.shape)==(1,1,256,256) and torch.isfinite(raw).all() and math.isfinite(score)
                native=torch.nn.functional.interpolate(raw,size=(1056,4224),mode='bilinear',align_corners=False)
                assert tuple(native.shape)==(1,1,1056,4224) and torch.isfinite(native).all()
                rn=raw[0,0].cpu().numpy(); nn=native[0,0].cpu().numpy(); sync(); et=time.perf_counter()-e0
                rp=OUT/'validation_maps'/f'{p.stem}_raw.npy'; npth=OUT/'validation_maps'/f'{p.stem}_native.npy'
                np.save(rp,rn); np.save(npth,nn)
                rows.append(dict(source=str(p),raw_image_score=score,model_map_max=float(rn.max()),native_map_max=float(nn.max()),
                  raw_map_shape=list(rn.shape),native_map_shape=list(nn.shape),finite=True,
                  raw_map=str(rp),native_map=str(npth),raw_sha256=sha(rp),native_sha256=sha(npth),
                  model_ms=mt*1000,end_to_end_ms=et*1000))
                print('VALIDATION',p.name,score,flush=True)
        infer_peak=peak(); assert c['val_forward_count']==4
        assert tensor_state_hash()==backbone_before
        stage='checkpoint serialization'
        checkpoint=OUT/'smoke_memory_bank.pt'
        torch.save(dict(state_dict=model.state_dict(),config=json.loads((OUT/'04_patchcore_config.json').read_text()),source_manifest_sha256=sha(OUT/'03_smoke_manifest.json')),checkpoint)
        loaded=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model.load_state_dict(loaded['state_dict'],strict=True)
        assert all(torch.equal(v.cpu(),loaded['state_dict'][k]) for k,v in model.state_dict().items())
        save('06_validation_forward.json',dict(count=4,rows=rows,metrics_computed=False,strict_state_reload='PASS',checkpoint_sha256=sha(checkpoint)))
        save('08_native_restore_audit.json',dict(native_restore_pass=True,model_map_shape=[256,256],native_shape=[1056,4224],count=4,nan=0,inf=0,interpolation='bilinear',align_corners=False,metric_source_sha256=hashes[str(ROOT/'src/evaluation/exact_aupro.py')]))
        definition='Maximum patch nearest-neighbor distance reweighted by softmax-derived neighborhood support (9 neighbors); official pred_score, not spatial max of smoothed map'
        save('07_image_score_audit.json',dict(official_image_score_available=True,definition=definition,rows=[{k:r[k] for k in ['source','raw_image_score','model_map_max','native_map_max']} for r in rows],score_replaced=False))
        mm=float(np.mean([r['model_ms'] for r in rows])); ee=float(np.mean([r['end_to_end_ms'] for r in rows]))
        runtime=dict(model_initialization_time_sec=c['model_initialization_time_sec'],train_feature_time_sec=ft,coreset_time_sec=ct,smoke_fit_time_sec=totalfit,
          validation_model_latency_ms=mm,validation_end_to_end_latency_ms=ee,validation_model_fps=1000/mm,
          feature_extraction_peak=feature_peak,coreset_peak=coreset_peak,validation_peak=infer_peak,
          warmup_extra_forwards=0,validation_samples=4,timing_note='Cold first sample included; end-to-end read + preprocessing + forward + native restore + CPU transfer, excludes npy saving/hash; 4 samples only',
          peak_cuda_allocated_mib=max(p['allocated_mib'] for p in [feature_peak,coreset_peak,infer_peak]),
          peak_cuda_reserved_mib=max(p['reserved_mib'] for p in [feature_peak,coreset_peak,infer_peak]))
        save('09_runtime_memory.json',runtime); c.update(runtime)
        fullraw=raw_count//16*137; fullbank=int(fullraw*0.1); ratio=137/16
        build=ft*ratio+ct*ratio**2
        # NN distance work scales with bank count, CNN cost does not; report broad bounds, never claim measured full cost.
        estimate=dict(estimate_label='ESTIMATED_NOT_MEASURED',estimated_full_raw_embedding_count=fullraw,
          estimated_full_coreset_count=fullbank,estimated_memory_bank_mib=fullbank*dim*4/2**20,
          estimated_full_raw_embeddings_mib=fullraw*dim*4/2**20,
          estimated_full_build_time=build,
          build_estimate_formula='feature time*(137/16) + coreset time*(137/16)^2; heuristic, projection dimension/logN and fixed overhead omitted',
          estimated_114_test_model_time=114*mm/1000,estimated_114_test_end_to_end_time=114*ee/1000,estimated_not_measured=True,
          estimated_full_test_inference_time=[114*ee/1000,114*(ee+mm*(ratio-1))/1000],
          inference_estimate_formula='lower: smoke E2E*114; upper: smoke E2E + model time*(137/16-1), times114; workload scenarios, not guaranteed bounds',
          full_run_8gb_feasible='Uncertain',reason='Smoke passes; raw bank ~822MiB and coreset ~82MiB exclude coreset projection/stack/workspace peaks; full fit not tested')
        save('10_full_run_estimate.json',estimate); c.update(estimate)
        assert all(sha(p)==v for p,v in hashes.items()) and not blocked
        assert all(sha(p)==v for p,v in previous_hashes.items()) and sha(wp)==weight_sha
        c.update(status='PASS_FOR_PATCHCORE_FORMAL',native_restore_pass=True,model_map_shape=[256,256],native_restore_shape=[1056,4224],
          patchcore_native_image_score_definition=definition,official_image_score_available=True,test_dependent_postprocessing=False,smoke_8gb_oom=False,phase20b_p2_patchcore_formal_authorized=True)
        save('12_phase20b_p2_preregistration.json',dict(dataset='MVTec AD 2',class_name='sheet_metal',train_normal=137,validation_normal=19,test_public=114,
          config={k:c[k] for k in ['backbone','layers','pre_trained','coreset_sampling_ratio','num_neighbors','precision','center_crop','seed']},
          input_hw=[256,256],pretrained_identifier=wa['pretrained_weight_identifier'],pretrained_sha256=weight_sha,
          post_processor=False,evaluator=False,visualizer=False,train_only_memory_bank=True,normalization_fitting=False,
          image_score=definition,native_restore='float32 bilinear align_corners=False to1056x4224',
          metrics=['Image AUROC','Pixel AUROC','Exact AUPRO@0.05','Exact AUPRO@0.30'],
          exact_aupro_sha256=hashes[str(ROOT/'src/evaluation/exact_aupro.py')],test_gt_used_for_tuning=False,
          formal_authorized=True,formal_started=False,OOM_policy='Stop without changing frozen config; authorization is not guarantee of full-scale8GB capacity',
          fresh_full_build_required=True,smoke_bank_not_formal=True))
    except Exception as e:
        c['error']=str(e); c['failed_stage']=stage
        if 'FAIL_WEIGHT_SHA_GATE' in str(e): c['status']='FAIL_WEIGHT_SHA_GATE'
        elif stage=='pretrained weights': c['status']='HOLD_PATCHCORE_PRETRAINED_WEIGHT'
        elif stage=='postprocessor': c['status']='HOLD_PATCHCORE_POSTPROCESSOR_PROTOCOL'
        elif stage in ['feature extraction','coreset']: c['status']='HOLD_PATCHCORE_MEMORY_BANK'
        elif 'out of memory' in str(e).lower(): c['status']='HOLD_PATCHCORE_8GB_OOM'
        if 'out of memory' in str(e).lower(): c.update(status='HOLD_PATCHCORE_8GB_OOM',smoke_8gb_oom=True)
        save('failure.json',dict(stage=stage,error=str(e),traceback=traceback.format_exc()))
        traceback.print_exc()
    finally:
        for key in ['pretrained_weight_source','raw_embedding_shape','raw_embedding_count','memory_bank_shape','memory_bank_count',
          'model_map_shape','native_restore_shape','patchcore_native_image_score_definition','peak_cuda_allocated_mib',
          'peak_cuda_reserved_mib','train_feature_time_sec','coreset_time_sec','smoke_fit_time_sec',
          'validation_model_latency_ms','validation_end_to_end_latency_ms','validation_model_fps',
          'estimated_full_build_time','estimated_full_test_inference_time','estimated_full_raw_embedding_count',
          'estimated_full_coreset_count','estimated_memory_bank_mib']:
            c.setdefault(key,None)
        for key in ['pretrained_weight_available','native_restore_pass','postprocessor_test_independent']:
            c.setdefault(key,False)
        c.setdefault('train_smoke_count',0); c.setdefault('validation_smoke_count',0)
        c.setdefault('full_run_8gb_feasible','Uncertain')
        save('11_test_leakage_guard.json',dict(test_public_files_accessed=0,test_mask_files_accessed=0,
          dataset_open_events=accessed,blocked_attempts=blocked,guard='Python audit hook rejects dataset access outside sheet_metal/train or validation, rejects bad/ground_truth',
          no_test_loader_created=True,metrics_computed=False))
        for name in ['02_weight_reverification.json','03_smoke_manifest.json','05_memory_bank_audit.json','06_validation_forward.json','08_native_restore_audit.json','07_image_score_audit.json','09_runtime_memory.json','10_full_run_estimate.json','12_phase20b_p2_preregistration.json']:
            if not (OUT/name).exists(): save(name,dict(status='NOT_EXECUTED',reason=c.get('error'),formal_authorized=False))
        save('completion_verification.json',c)
        (OUT/'final_console_report.txt').write_text('PHASE20B-P1 PATCHCORE TEST-FREE SMOKE\n'+json.dumps(c,indent=2,ensure_ascii=False)+'\nNo formal run, FastFlow or test access performed.\n',encoding='utf8')
        print(json.dumps(c,indent=2,ensure_ascii=False),flush=True)
    return 0 if c['status']=='PASS_FOR_PATCHCORE_FORMAL' else 1
if __name__=='__main__': sys.exit(main())
