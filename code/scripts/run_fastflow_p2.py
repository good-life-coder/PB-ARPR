"""Phase20C-P1: official FastFlow, normal-only smoke; no formal training or metrics."""
import os,sys,json,time,hashlib,math,traceback,platform,inspect
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PRE=ROOT/'experiments/phase20c_p1_fastflow_smoke'
OUT=ROOT/'experiments/phase20c_p2_fastflow_formal'
TEST_UNLOCKED=False
DATA=ROOT/'datasets/MVTec_AD_2/sheet_metal'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(n,d): (OUT/n).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def allowed(p):
    p=Path(p).resolve(); rel=p.relative_to(DATA.resolve())
    if TEST_UNLOCKED and rel.parts and rel.parts[0]=='test_public':return p
    if len(rel.parts)<2 or rel.parts[0] not in ('train','validation') or rel.parts[1]!='good': raise PermissionError(str(p))
    return p
def main():
    global TEST_UNLOCKED
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
    c=dict(phase='Phase20C-P2',status='HOLD_FASTFLOW_PROTOCOL',backbone='resnet18',pretrained=True,pretrained_tag='a1_in1k',
      input_height=256,input_width=256,center_crop=None,flow_steps=8,hidden_ratio=1.0,conv3x3_only=False,precision='float32',seed=42,
      optimizer='Adam',learning_rate=0.001,weight_decay=0.00001,scheduler=None,batch_size=32,
      train_smoke_count=16,validation_smoke_count=4,smoke_steps=20,forward_count=0,backward_count=0,optimizer_step_count=0,
      val_forward_count=0,grad_nan_count=0,grad_inf_count=0,test_public_files_accessed=0,test_mask_files_accessed=0,
      test_dependent_normalization=False,test_dependent_threshold=False,test_dependent_calibration=False,
      phase20c_p2_fastflow_formal_authorized=False,protocol_violation_count=0)
    stage='previous gate'
    try:
        pg=json.loads((PRE/'completion_verification.json').read_text())
        (PRE/'final_console_report.txt').read_text(encoding='utf8')
        assert pg['status']=='PASS_FOR_FASTFLOW_FORMAL' and pg['phase20c_p2_fastflow_formal_authorized'],'FAIL_PHASE20C_P1_GATE'
        assert pg['test_public_files_accessed']==pg['test_mask_files_accessed']==pg['protocol_violation_count']==0
        for k in ['backbone','pretrained_tag','input_height','input_width','flow_steps','hidden_ratio','conv3x3_only','optimizer','learning_rate','weight_decay','scheduler','batch_size','precision','seed']:
            assert c[k]==pg[k], 'Frozen config mismatch '+k
        pa=ROOT/'experiments/phase20a_external_baseline_audit'; a=json.loads((pa/'completion_verification.json').read_text())
        (pa/'final_console_report.txt').read_text(encoding='utf8')
        assert a['status']=='PASS_FOR_EXTERNAL_BASELINE_SMOKE' and a['fastflow_available'] and a['anomalib_version']=='2.6.1','FAIL_PHASE20A_GATE'
        prior=[PRE/'completion_verification.json',PRE/'final_console_report.txt',pa/'completion_verification.json',pa/'final_console_report.txt',ROOT/'experiments/phase20b_p2_patchcore_formal/completion_verification.json']
        historical={str(p):sha(p) for p in prior}
        save('01_preflight.json',dict(status='PASS',sources=historical))
        stage='weights'
        meta=json.loads((PRE/'hf_metadata.json').read_text()); wi=next(x for x in meta['siblings'] if x['rfilename']=='model.safetensors')
        wp=PRE/'model.safetensors'; assert wp.stat().st_size==wi['size'] and sha(wp)==wi['lfs']['sha256']
        c['weight_sha256']=sha(wp)
        assert c['weight_sha256']==pg['weight_sha256'],'FAIL_WEIGHT_GATE'
        cache=PRE/'weights/hub'; repo=cache/'models--timm--resnet18.a1_in1k'; commit=meta['sha']
        dest=repo/'snapshots'/commit/'model.safetensors';dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():os.link(wp,dest)
        assert sha(dest)==c['weight_sha256']
        (repo/'refs').mkdir(exist_ok=True); assert (repo/'refs/main').read_text()==commit
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
        reference=(PRE/'official_fastflow_v070.yaml').read_text()
        assert 'max_epochs: 500' in reference and 'train_batch_size: 32' in reference
        source='https://github.com/open-edge-platform/anomalib/blob/v0.7.0/src/anomalib/models/fastflow/config.yaml'
        budget=dict(proposed_formal_epochs=500,proposed_formal_max_steps=2500,trainer_max_steps=-1,steps_per_epoch=5,batch_size=32,drop_last=False,
          formal_budget_source=source,reference_sha256=sha(PRE/'official_fastflow_v070.yaml'),
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
        save('02_fastflow_config.json',dict(c,transform=repr(model.pre_processor.transform),post_processor=False,evaluator=False,visualizer=False,optimizer_defaults={k:str(v) for k,v in optimizer.defaults.items()}))
        stage='manifest';tr=sorted((DATA/'train/good').glob('*.png'));va=sorted((DATA/'validation/good').glob('*.png'));assert len(tr)==137 and len(va)==19
        train=tr;val=va
        save('04_train_manifest.json',dict(train=[dict(path=str(p),sha256=sha(p)) for p in train],validation=[dict(path=str(p),sha256=sha(p)) for p in val],
          selection='All frozen137 train normal and19 validation normal',formal_batch_rule='Seed42 epoch shuffles; batch32; drop_last=False; final batch9; workers0',formal_train_total=137,formal_validation_total=19))
        def load(p):return model.pre_processor.transform(read_image(allowed(p),as_tensor=True))
        c.update(formal_epochs=500,final_epoch=0,actual_total_steps=0,early_stopping_enabled=False,train_count=137,validation_count=19,
          loss_nan_count=0,loss_inf_count=0,dataset='MVTec AD 2',class_name='sheet_metal',fresh_initialization=True,smoke_checkpoint_used=False)
        for k in ['train_smoke_count','validation_smoke_count','smoke_steps','phase20c_p2_fastflow_formal_authorized']:
            c.pop(k,None)
        save('03_weight_verification.json',dict(path=str(wp),identifier='timm/resnet18.a1_in1k',sha256=c['weight_sha256'],match=True,strict_full_load='PASS'))
        def state_hash(module):
            h=hashlib.sha256()
            for key,v in module.state_dict().items():h.update(key.encode());h.update(v.detach().cpu().numpy().tobytes())
            return h.hexdigest()
        c['fresh_flow_hash']=state_hash(model.model.fast_flow_blocks)
        backbone_hash=state_hash(model.model.feature_extractor)
        save('initialization.json',dict(backbone_sha256=backbone_hash,flow_sha256=c['fresh_flow_hash'],seed=42,fresh=True))
        stage='formal training';model.train();torch.cuda.reset_peak_memory_stats();losses=[];step_times=[]
        generator=torch.Generator().manual_seed(42)
        log=(OUT/'training_history.csv').open('w',encoding='utf8',buffering=1)
        log.write('epoch,global_step,loss,optimizer_step_sec,source_ids\n')
        torch.cuda.synchronize();training_start=time.perf_counter()
        for epoch in range(1,501):
            order=torch.randperm(len(train),generator=generator).tolist()
            for offset in range(0,len(order),32):
                ids=order[offset:offset+32]
                x=torch.stack([load(train[i]) for i in ids]).cuda()
                torch.cuda.synchronize();t=time.perf_counter();optimizer.zero_grad(set_to_none=True)
                hidden,jac=model.model(x);c['forward_count']+=1;loss=model.loss(hidden,jac)
                c['loss_nan_count']+=int(torch.isnan(loss));c['loss_inf_count']+=int(torch.isinf(loss))
                assert c['loss_nan_count']==c['loss_inf_count']==0
                loss.backward();c['backward_count']+=1
                for p in model.parameters():
                    if p.grad is not None:
                        c['grad_nan_count']+=int(torch.isnan(p.grad).sum());c['grad_inf_count']+=int(torch.isinf(p.grad).sum())
                assert c['grad_nan_count']==c['grad_inf_count']==0
                optimizer.step();c['optimizer_step_count']+=1;c['actual_total_steps']+=1
                torch.cuda.synchronize();step_times.append(time.perf_counter()-t);losses.append(float(loss.item()))
                log.write(f"{epoch},{c['actual_total_steps']},{losses[-1]},{step_times[-1]},"+';'.join(train[i].name for i in ids)+'\n')
            c['final_epoch']=epoch
            elapsed=time.perf_counter()-training_start
            save('progress.json',dict(epoch=epoch,target_epochs=500,steps=c['actual_total_steps'],loss=losses[-1],elapsed_sec=elapsed,estimated_remaining_sec=elapsed/epoch*(500-epoch),test_public_files_accessed=0))
            if epoch%10==0:print('EPOCH',epoch,'STEPS',c['actual_total_steps'],'ELAPSED_SEC',round(elapsed,2),flush=True)
            if epoch%25==0:
                torch.save(dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,global_step=c['actual_total_steps'],config=c,generator_state=generator.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),OUT/'last.ckpt')
        torch.cuda.synchronize();log.close()
        c.update(training_wall_time_sec=time.perf_counter()-training_start,mean_optimizer_step_sec=float(np.mean(step_times)),median_optimizer_step_sec=float(np.median(step_times)),
          train_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,train_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
          first_train_loss=losses[0],final_train_loss=losses[-1],min_train_loss=min(losses),max_train_loss=max(losses),steps_per_epoch=5)
        c['training_wall_time_hours']=c['training_wall_time_sec']/3600
        assert c['final_epoch']==500 and c['actual_total_steps']==c['forward_count']==c['backward_count']==c['optimizer_step_count']==2500
        assert state_hash(model.model.feature_extractor)==backbone_hash
        assert state_hash(model.model.fast_flow_blocks)!=c['fresh_flow_hash']
        assert all(torch.isfinite(p).all() for p in model.parameters())
        save('05_training_summary.json',dict(c,status='PASS',test_public_files_accessed=0,test_mask_files_accessed=0))
        save('06_training_numerical_audit.json',{k:c[k] for k in ['loss_nan_count','loss_inf_count','grad_nan_count','grad_inf_count','first_train_loss','final_train_loss','min_train_loss','max_train_loss']})
        stage='checkpoint';checkpoint=OUT/'final_500_epochs.ckpt'
        torch.save(dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),epoch=500,global_step=2500,config=c),checkpoint)
        loaded=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model.load_state_dict(loaded['state_dict'],strict=True)
        assert all(torch.equal(v.cpu(),loaded['state_dict'][k]) for k,v in model.state_dict().items())
        c.update(checkpoint_path=str(checkpoint),checkpoint_sha256=sha(checkpoint),strict_reload='PASS',checkpoint_frozen_timestamp=time.time())
        save('07_checkpoint_verification.json',dict(path=str(checkpoint),sha256=c['checkpoint_sha256'],epoch=500,optimizer_steps=2500,strict_reload='PASS',frozen_timestamp=c['checkpoint_frozen_timestamp'],test_public_files_accessed=0))
        del loaded,x,hidden,jac,loss;optimizer.zero_grad(set_to_none=True)
        TEST_UNLOCKED=True;stage='formal evaluation'
        from fastflow_p2_evaluation import finish
        def load_batch(p):return load(p)[None].cuda()
        def sync():torch.cuda.synchronize()
        def peak():return dict(allocated_mib=torch.cuda.max_memory_allocated()/2**20,reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        finish(model,c,OUT,ROOT,DATA,load_batch,sync,peak,sha,save)
        assert all(sha(p)==h for p,h in historical.items()) and all(sha(p)==h for p,h in sourcehash.items()) and not blocked
        c['status']='PASS'
    except Exception as e:
        c.update(error=str(e),failed_stage=stage,status='FAIL_FASTFLOW_FORMAL')
        if 'out of memory' in str(e).lower():c['status']='FAIL_FASTFLOW_FORMAL_OOM'
        traceback.print_exc()
    finally:
        c['protocol_violation_count']=len(blocked)
        save('17_protocol_audit.json',dict(accessed=accessed,blocked=blocked,protocol_violation_count=len(blocked),test_unlocked=TEST_UNLOCKED,
          training_test_public_files_accessed=0,training_test_mask_files_accessed=0,test_dependent_normalization=False,test_dependent_threshold=False,test_dependent_calibration=False))
        save('completion_verification.json',c)
        (OUT/'final_console_report.txt').write_text('PHASE20C-P2 FASTFLOW FORMAL EXTERNAL BASELINE\n'+json.dumps(c,indent=2,ensure_ascii=False)+'\nNo other baseline/seed or parameter search started.\n',encoding='utf8')
        print(json.dumps(c,indent=2),flush=True)
    return 0 if c['status']=='PASS' else 1
if __name__=='__main__':sys.exit(main())
