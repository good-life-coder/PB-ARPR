"""Phase20C-P1: official FastFlow, normal-only smoke; no formal training or metrics."""
import os,sys,json,time,hashlib,math,traceback,platform,inspect
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/phase20c_p1_fastflow_smoke'
DATA=ROOT/'datasets/MVTec_AD_2/sheet_metal'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(n,d): (OUT/n).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def allowed(p):
    p=Path(p).resolve(); rel=p.relative_to(DATA.resolve())
    if len(rel.parts)<2 or rel.parts[0] not in ('train','validation') or rel.parts[1]!='good': raise PermissionError(str(p))
    return p
def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'completion_verification.json').exists(),'Previous result exists; do not rerun'
    accessed=[];blocked=[]
    def guard(event,args):
        if event=='socket.connect': raise PermissionError('Smoke must be offline')
        if event not in ('open','os.listdir','os.scandir') or not args or not isinstance(args[0],(str,bytes)):return
        p=Path(os.fsdecode(args[0])).absolute()
        if p.is_relative_to(ROOT/'datasets'):
            try: allowed(p)
            except Exception: blocked.append(str(p));raise
            if event=='open':accessed.append(str(p))
    sys.addaudithook(guard)
    c=dict(phase='Phase20C-P1',status='HOLD_FASTFLOW_PROTOCOL',backbone='resnet18',pretrained=True,pretrained_tag='a1_in1k',
      input_height=256,input_width=256,center_crop=None,flow_steps=8,hidden_ratio=1.0,conv3x3_only=False,precision='float32',seed=42,
      optimizer='Adam',learning_rate=0.001,weight_decay=0.00001,scheduler=None,batch_size=32,
      train_smoke_count=16,validation_smoke_count=4,smoke_steps=20,forward_count=0,backward_count=0,optimizer_step_count=0,
      val_forward_count=0,grad_nan_count=0,grad_inf_count=0,test_public_files_accessed=0,test_mask_files_accessed=0,
      test_dependent_normalization=False,test_dependent_threshold=False,test_dependent_calibration=False,
      phase20c_p2_fastflow_formal_authorized=False,protocol_violation_count=0)
    stage='previous gate'
    try:
        pa=ROOT/'experiments/phase20a_external_baseline_audit'; a=json.loads((pa/'completion_verification.json').read_text())
        (pa/'final_console_report.txt').read_text(encoding='utf8')
        assert a['status']=='PASS_FOR_EXTERNAL_BASELINE_SMOKE' and a['fastflow_available'] and a['anomalib_version']=='2.6.1','FAIL_PHASE20A_GATE'
        prior=[pa/'completion_verification.json',pa/'final_console_report.txt',ROOT/'experiments/phase20b_p2_patchcore_formal/completion_verification.json']
        historical={str(p):sha(p) for p in prior}
        save('01_phase20a_gate.json',dict(status='PASS',sources=historical))
        stage='weights'
        meta=json.loads((OUT/'hf_metadata.json').read_text()); wi=next(x for x in meta['siblings'] if x['rfilename']=='model.safetensors')
        wp=OUT/'model.safetensors'; assert wp.stat().st_size==wi['size'] and sha(wp)==wi['lfs']['sha256']
        c['weight_sha256']=sha(wp)
        cache=OUT/'weights/hub'; repo=cache/'models--timm--resnet18.a1_in1k'; commit=meta['sha']
        dest=repo/'snapshots'/commit/'model.safetensors';dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():os.link(wp,dest)
        assert sha(dest)==c['weight_sha256']
        (repo/'refs').mkdir(exist_ok=True); (repo/'refs/main').write_text(commit)
        os.environ['HF_HUB_CACHE']=str(cache);os.environ['HF_HUB_OFFLINE']='1'
        import torch,numpy as np,timm,torchvision,importlib.metadata as md
        from safetensors.torch import load_file
        from anomalib.models import Fastflow
        from anomalib.data.utils import read_image
        from lightning import seed_everything
        assert md.version('anomalib')=='2.6.1'
        env=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,torchvision=torchvision.__version__,anomalib=md.version('anomalib'),timm=timm.__version__,gpu=torch.cuda.get_device_name())
        save('02_environment.json',env);c.update(anomalib_version=env['anomalib'],fastflow_source=inspect.getfile(Fastflow))
        # Current implementation determines model/optimizer. Historical official recipe supplies budget only.
        reference=(OUT/'official_fastflow_v070.yaml').read_text()
        assert 'max_epochs: 500' in reference and 'train_batch_size: 32' in reference
        source='https://github.com/open-edge-platform/anomalib/blob/v0.7.0/src/anomalib/models/fastflow/config.yaml'
        budget=dict(proposed_formal_epochs=500,proposed_formal_max_steps=2500,trainer_max_steps=-1,steps_per_epoch=5,batch_size=32,drop_last=False,
          formal_budget_source=source,reference_sha256=sha(OUT/'official_fastflow_v070.yaml'),
          derivation='500 epochs * ceil(137/32) = 2500 optimizer steps; final batch9; epoch-controlled; reference budget, not 2.6.1 class default',
          differences='Current 2.6.1 conv3x3_only=False retained (old recipe True); old early stopping/min-max/test-derived validation disabled; fixed final epoch only.',
          early_stopping=False,checkpoint_selection='final epoch500 only',test_gt_tuning=False,seed=42,shuffle=True,precision='float32',fresh_initialization=True)
        save('14_formal_budget_audit.json',budget);c.update({k:budget[k] for k in ['proposed_formal_epochs','proposed_formal_max_steps','formal_budget_source']})
        stage='model initialization';t=time.perf_counter();seed_everything(42,workers=True)
        model=Fastflow(backbone='resnet18',pre_trained=True,flow_steps=8,hidden_ratio=1.0,conv3x3_only=False,
          pre_processor=Fastflow.configure_pre_processor((256,256)),post_processor=False,evaluator=False,visualizer=False)
        weights=load_file(str(wp));full=timm.create_model('resnet18',pretrained=False)
        cfg=full.pretrained_cfg; assert cfg['tag']=='a1_in1k' and cfg['hf_hub_id']=='timm/resnet18.a1_in1k'
        result=full.load_state_dict(weights,strict=True);assert not result.missing_keys and not result.unexpected_keys
        for key,v in model.model.feature_extractor.state_dict().items():
            wk=key if key in weights else key.replace('_','.')
            assert wk in weights and torch.equal(v.cpu(),weights[wk]),key
        del full,weights
        c.update(parameter_count=sum(p.numel() for p in model.parameters()),trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad))
        assert all(torch.isfinite(p).all() for p in model.parameters())
        stage='postprocessor';assert model.post_processor is None and model.evaluator is None and model.visualizer is None
        assert 'CenterCrop' not in repr(model.pre_processor.transform)
        model=model.cuda();torch.cuda.synchronize();c['model_init_sec']=time.perf_counter()-t
        optimizer=model.configure_optimizers();assert isinstance(optimizer,torch.optim.Adam)
        assert optimizer.param_groups[0]['lr']==0.001 and optimizer.param_groups[0]['weight_decay']==1e-5
        files=[Path(inspect.getfile(Fastflow)),Path(inspect.getfile(type(model.model))),Path(inspect.getfile(type(model.loss))),Path(inspect.getfile(type(model.model.anomaly_map_generator)))]
        sourcehash={str(p):sha(p) for p in files}
        save('03_fastflow_implementation_audit.json',dict(class_path='anomalib.models.image.fastflow.lightning_model.Fastflow',constructor=str(inspect.signature(Fastflow)),
          pretrained_cfg=cfg,trainer_arguments=model.trainer_arguments,default_epochs=None,default_batch_size=None,reference_epochs=500,reference_batch_size=32,sources=sourcehash,
          default_engine_budget='Not method-specific; not adopted',flow='3 feature scales,8 coupling blocks each,alternating3x3/1x1,affine clamp2',precision='explicitFP32'))
        save('04_fastflow_config.json',dict(c,transform=repr(model.pre_processor.transform),post_processor=False,evaluator=False,visualizer=False,optimizer_defaults={k:str(v) for k,v in optimizer.defaults.items()}))
        stage='manifest';tr=sorted((DATA/'train/good').glob('*.png'));va=sorted((DATA/'validation/good').glob('*.png'));assert len(tr)==137 and len(va)==19
        train=tr[:16];val=va[:4]
        save('07_smoke_manifest.json',dict(train=[dict(path=str(p),sha256=sha(p)) for p in train],validation=[dict(path=str(p),sha256=sha(p)) for p in val],
          selection='Stable lexicographic first16/first4',smoke_batch_rule='Each batch32 cycles the16 selected normals twice; no other sources; tests formal full-batch memory',formal_train_total=137,formal_validation_total=19))
        def load(p):return model.pre_processor.transform(read_image(allowed(p),as_tensor=True))
        xs=torch.stack([load(p) for p in train])
        stage='feature sanity'
        with torch.no_grad():
            features=model.model.feature_extractor.eval()(xs[:1].cuda());assert all(torch.isfinite(f).all() for f in features)
        save('05_pretrained_weight_audit.json',dict(status='PASS',path=str(wp),size=wp.stat().st_size,sha256=c['weight_sha256'],official_repo=meta['id'],revision=commit,
          source='https://huggingface.co/'+meta['id']+'/resolve/'+commit+'/model.safetensors',strict_full_model_load='PASS',feature_extractor_exact_weights_match=True,feature_shapes=[list(f.shape) for f in features],feature_finite=True))
        del features
        stage='training';losses=[];times=[];model.train();torch.cuda.reset_peak_memory_stats()
        before={k:v.detach().cpu().clone() for k,v in model.model.feature_extractor.state_dict().items()}
        flow_before=next(model.model.fast_flow_blocks.parameters()).detach().cpu().clone()
        for i in range(20):
            torch.cuda.synchronize();start=time.perf_counter();x=xs.repeat(2,1,1,1).cuda();optimizer.zero_grad(set_to_none=True)
            hidden,jac=model.model(x);c['forward_count']+=1;loss=model.loss(hidden,jac);assert torch.isfinite(loss)
            loss.backward();c['backward_count']+=1
            for p in model.parameters():
                if p.grad is not None:
                    c['grad_nan_count']+=int(torch.isnan(p.grad).sum());c['grad_inf_count']+=int(torch.isinf(p.grad).sum())
            assert c['grad_nan_count']==c['grad_inf_count']==0
            optimizer.step();c['optimizer_step_count']+=1;torch.cuda.synchronize();times.append(time.perf_counter()-start);losses.append(float(loss.item()))
            print('SMOKE_STEP',i+1,losses[-1],times[-1],flush=True)
        assert all(torch.equal(before[k],v.cpu()) for k,v in model.model.feature_extractor.state_dict().items())
        assert not torch.equal(flow_before,next(model.model.fast_flow_blocks.parameters()).detach().cpu())
        c.update(first_loss=losses[0],last_loss=losses[-1],min_loss=min(losses),max_loss=max(losses),mean_train_step_sec=float(np.mean(times[3:])),median_train_step_sec=float(np.median(times[3:])),
          train_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,train_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        save('08_training_smoke.json',dict(steps=20,losses=losses,times_sec=times,batch_size=32,unique_train_sources=16,backbone_unchanged=True,flow_updated=True,first3_steps_excluded_from_speed=True))
        save('09_numerical_stability.json',dict(loss_finite=True,grad_nan_count=0,grad_inf_count=0,oom=False))
        del x,hidden,jac,loss;optimizer.zero_grad(set_to_none=True)
        stage='validation';model.eval();torch.cuda.reset_peak_memory_stats();rows=[];(OUT/'validation_maps').mkdir(exist_ok=True)
        with torch.no_grad():
            for p in val:
                torch.cuda.synchronize();t=time.perf_counter();x=load(p)[None].cuda();torch.cuda.synchronize();m=time.perf_counter()
                out=model.model(x);torch.cuda.synchronize();mt=time.perf_counter()-m;c['val_forward_count']+=1
                raw=out.anomaly_map.float();score=float(out.pred_score.item());assert torch.isfinite(raw).all() and math.isfinite(score)
                assert torch.equal(out.pred_score,raw.amax(dim=(-2,-1)))
                rt=time.perf_counter();native=torch.nn.functional.interpolate(raw,size=(1056,4224),mode='bilinear',align_corners=False);torch.cuda.synchronize();rt=time.perf_counter()-rt
                rn=raw[0,0].cpu().numpy();nn=native[0,0].cpu().numpy();assert rn.shape==(256,256) and nn.shape==(1056,4224) and np.isfinite(nn).all()
                et=time.perf_counter()-t;rp=OUT/'validation_maps'/f'{p.stem}_raw.npy';npth=OUT/'validation_maps'/f'{p.stem}_native.npy';np.save(rp,rn);np.save(npth,nn)
                rows.append(dict(source=str(p),official_image_score=score,model_map_max=float(rn.max()),model_ms=mt*1000,end_to_end_ms=et*1000,restore_ms=rt*1000,raw_path=str(rp),native_path=str(npth),raw_sha256=sha(rp),native_sha256=sha(npth)))
        definition='Official spatial maximum of 256x256 model anomaly map, before native resize. Map: mean over scales of bilinear-upsampled negative exp(-0.5*mean_channels(z^2)); negative values retained.'
        c.update(fastflow_image_score_definition=definition,official_image_score_available=True,model_map_shape=[256,256],native_restore_shape=[1056,4224],native_restore_pass=True,
          inference_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,inference_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
          model_only_inference_ms=float(np.mean([r['model_ms'] for r in rows])),end_to_end_inference_ms=float(np.mean([r['end_to_end_ms'] for r in rows])))
        c['model_only_fps']=1000/c['model_only_inference_ms'];c['estimated_formal_training_hours']=2500*c['mean_train_step_sec']/3600
        c['estimated_114_test_time']=114*c['model_only_inference_ms']/1000;c['full_run_8gb_feasible']=True
        save('10_validation_forward.json',dict(count=4,rows=rows,metrics_computed=False))
        save('11_image_score_audit.json',dict(definition=definition,official_score_available=True,score_equals_model_map_max=True,score_replaced=False))
        save('12_native_restore_audit.json',dict(count=4,native_shape=[1056,4224],dtype='float32',interpolation='bilinear',align_corners=False,nan=0,inf=0,status='PASS'))
        save('13_runtime_memory.json',dict(c,timing_note='Training excludes first3 steps, includes finite-gradient diagnostics and H2D but preprocessed inputs cached. Validation cold-inclusive4 images; E2E excludes npy writes/hash. Reserved VRAM retains training cache.'))
        save('15_formal_runtime_estimate.json',dict(estimated_not_measured=True,training_hours=c['estimated_formal_training_hours'],model_inference114_sec=c['estimated_114_test_time'],
          native_restore114_sec=114*np.mean([r['restore_ms'] for r in rows])/1000,end_to_end114_sec=114*c['end_to_end_inference_ms']/1000,
          caveat='137/32 produces4 full batches and1 batch9; estimate charges all5 as full batches. Smoke diagnostics inflate timing; real disk loading/checkpoint/validation overhead differs.'))
        assert all(sha(p)==h for p,h in historical.items()) and all(sha(p)==h for p,h in sourcehash.items()) and not blocked
        c.update(status='PASS_FOR_FASTFLOW_FORMAL',phase20c_p2_fastflow_formal_authorized=True)
        save('16_phase20c_p2_preregistration.json',dict(c,budget=budget,train_count=137,validation_count=19,test_count=114,formal_started=False,
          formal_fresh_required=True,smoke_weights_discarded=True,final_epoch_only=True,no_test_tuning=True,metrics=['Image AUROC','Pixel AUROC','Exact AUPRO005','Exact AUPRO030'],
          exact_aupro_sha256=sha(ROOT/'src/evaluation/exact_aupro.py'),native_restore='float32 bilinear align_corners=False1056x4224',image_score='official model-grid spatial max',source_sha256=sourcehash))
    except Exception as e:
        c.update(error=str(e),failed_stage=stage)
        if 'out of memory' in str(e).lower():c['status']='HOLD_FASTFLOW_8GB_OOM'
        elif stage in ('weights','model initialization'):c['status']='HOLD_FASTFLOW_PRETRAINED_WEIGHT'
        elif stage=='training':c['status']='HOLD_FASTFLOW_NUMERICAL'
        elif stage=='postprocessor':c['status']='HOLD_FASTFLOW_POSTPROCESSOR'
        if 'torch' in locals() and torch.cuda.is_available():c.update(failure_allocated_mib=torch.cuda.max_memory_allocated()/2**20,failure_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        traceback.print_exc()
    finally:
        c['protocol_violation_count']=len(blocked)
        save('06_test_leakage_guard.json',dict(test_public_files_accessed=0,test_mask_files_accessed=0,accessed=accessed,blocked=blocked,policy='Only sheet_metal/train/good and validation/good may be opened; all network blocked during smoke'))
        names=['01_phase20a_gate','02_environment','03_fastflow_implementation_audit','04_fastflow_config','05_pretrained_weight_audit','07_smoke_manifest','08_training_smoke','09_numerical_stability','10_validation_forward','11_image_score_audit','12_native_restore_audit','13_runtime_memory','14_formal_budget_audit','15_formal_runtime_estimate','16_phase20c_p2_preregistration']
        for n in names:
            if not (OUT/(n+'.json')).exists():save(n+'.json',dict(status='NOT_EXECUTED',reason=c.get('error')))
        save('completion_verification.json',c);(OUT/'final_console_report.txt').write_text('PHASE20C-P1 FASTFLOW TEST-FREE SMOKE\n'+json.dumps(c,indent=2,ensure_ascii=False)+'\nFormal training NOT started.\n',encoding='utf8')
        print(json.dumps(c,indent=2),flush=True)
    return 0 if c['status']=='PASS_FOR_FASTFLOW_FORMAL' else 1
if __name__=='__main__':sys.exit(main())
