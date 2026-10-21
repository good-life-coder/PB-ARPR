"""Derive formal orchestration from P1 and frozen P2 external evaluation; algorithms unchanged."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'scripts/smoke_fastflow_p1.py').read_text(encoding='utf8')
s=s[:s.index('        xs=torch.stack')]
s=s.replace("OUT=ROOT/'experiments/phase20c_p1_fastflow_smoke'", "PRE=ROOT/'experiments/phase20c_p1_fastflow_smoke'\nOUT=ROOT/'experiments/phase20c_p2_fastflow_formal'\nTEST_UNLOCKED=False")
s=s.replace("    if len(rel.parts)<2", "    if TEST_UNLOCKED and rel.parts and rel.parts[0]=='test_public':return p\n    if len(rel.parts)<2")
s=s.replace('    OUT.mkdir(exist_ok=True)', '    global TEST_UNLOCKED\n    OUT.mkdir(exist_ok=True)')
s=s.replace("phase='Phase20C-P1'", "phase='Phase20C-P2'")
s=s.replace("        pa=ROOT/", "        pg=json.loads((PRE/'completion_verification.json').read_text())\n        (PRE/'final_console_report.txt').read_text(encoding='utf8')\n        assert pg['status']=='PASS_FOR_FASTFLOW_FORMAL' and pg['phase20c_p2_fastflow_formal_authorized'],'FAIL_PHASE20C_P1_GATE'\n        assert pg['test_public_files_accessed']==pg['test_mask_files_accessed']==pg['protocol_violation_count']==0\n        for k in ['backbone','pretrained_tag','input_height','input_width','flow_steps','hidden_ratio','conv3x3_only','optimizer','learning_rate','weight_decay','scheduler','batch_size','precision','seed']:\n            assert c[k]==pg[k], 'Frozen config mismatch '+k\n        pa=ROOT/",1)
s=s.replace("prior=[pa/", "prior=[PRE/'completion_verification.json',PRE/'final_console_report.txt',pa/")
s=s.replace("OUT/'hf_metadata.json'", "PRE/'hf_metadata.json'").replace("wp=OUT/'model.safetensors'", "wp=PRE/'model.safetensors'")
s=s.replace("        c['weight_sha256']=sha(wp)", "        c['weight_sha256']=sha(wp)\n        assert c['weight_sha256']==pg['weight_sha256'],'FAIL_WEIGHT_GATE'")
s=s.replace("cache=OUT/'weights/hub'", "cache=PRE/'weights/hub'")
s=s.replace("(repo/'refs/main').write_text(commit)", "assert (repo/'refs/main').read_text()==commit")
s=s.replace("OUT/'official_fastflow_v070.yaml'", "PRE/'official_fastflow_v070.yaml'")
s=s.replace('train=tr[:16];val=va[:4]','train=tr;val=va')
s=s.replace("selection='Stable lexicographic first16/first4'", "selection='All frozen137 train normal and19 validation normal'")
s=s.replace("smoke_batch_rule='Each batch32 cycles the16 selected normals twice; no other sources; tests formal full-batch memory'", "formal_batch_rule='Seed42 epoch shuffles; batch32; drop_last=False; final batch9; workers0'")
s=s.replace("'01_phase20a_gate.json'", "'01_preflight.json'").replace("'04_fastflow_config.json'", "'02_fastflow_config.json'").replace("'07_smoke_manifest.json'", "'04_train_manifest.json'")
s+='''        c.update(formal_epochs=500,final_epoch=0,actual_total_steps=0,early_stopping_enabled=False,train_count=137,validation_count=19,
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
        log.write('epoch,global_step,loss,optimizer_step_sec,source_ids\\n')
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
                log.write(f"{epoch},{c['actual_total_steps']},{losses[-1]},{step_times[-1]},"+';'.join(train[i].name for i in ids)+'\\n')
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
        (OUT/'final_console_report.txt').write_text('PHASE20C-P2 FASTFLOW FORMAL EXTERNAL BASELINE\\n'+json.dumps(c,indent=2,ensure_ascii=False)+'\\nNo other baseline/seed or parameter search started.\\n',encoding='utf8')
        print(json.dumps(c,indent=2),flush=True)
    return 0 if c['status']=='PASS' else 1
if __name__=='__main__':sys.exit(main())
'''
(ROOT/'scripts/run_fastflow_p2.py').write_text(s,encoding='utf8')
e=(ROOT/'scripts/patchcore_p2_evaluation.py').read_text(encoding='utf8')
e=e.replace("weight_sha256=c['pretrained_weight_sha256'],pretrained_tag='racm_in1k'", "pretrained_tag='a1_in1k'")
e=e.replace("patchcore_image_score_definition='OFFICIAL_PATCHCORE_WEIGHTED_NN_SCORE'", "fastflow_image_score_definition='MODEL_GRID_SPATIAL_MAX'")
e=e.replace("c['patchcore_image_score_definition']", "c['fastflow_image_score_definition']")
e=e.replace("Maximum anomalous patch nearest-neighbor distance with 9-neighbor support weighting; official pred_score retained", "Official maximum of model-grid256x256 anomaly map before native restore; raw negative scores retained")
e=e.replace("            assert raw.shape==", "            assert torch.equal(prediction.pred_score,raw.amax(dim=(-2,-1)))\n            assert raw.shape==")
e=e.replace("'PatchCore':{k:c[k]", "'FastFlow':{k:c[k]")
e=e[:e.index('    table=[')]+'''    patch=read('experiments/phase20b_p2_patchcore_formal/completion_verification.json')
    for key,name in zip(keys,['image','pixel','aupro005','aupro030']):c[f'delta_vs_patchcore_{name}_pp']=100*(c[key]-patch[key])
    comparison['PatchCore']={k:patch[k] for k in keys}
    save('15_comparison_with_external_baselines.json',dict(comparison,deltas={k:v for k,v in c.items() if k.startswith('delta_')}))
    table=read('experiments/phase20b_p2_patchcore_formal/15_external_comparison_table.json')
    table=[r for r in table if r['method']!='FastFlow']
    table.append(dict(method='FastFlow',backbone='ResNet18 a1_in1k',input=[256,256],training_strategy='normal-only flows and LayerNorm; frozen backbone',training_budget='500epochs/batch32/2500steps',build_train_sec=c['training_wall_time_sec'],model_fps=c['model_only_fps'],peak_vram_mib=c['train_peak_allocated_mib'],inference_peak_vram_mib=c['inference_peak_allocated_mib'],**comparison['FastFlow']))
    save('16_external_comparison_table.json',table)
    save('11_evaluation_report.json',c)
'''
for a,b in [('07_test_inference_summary','08_test_inference_summary'),('08_image_score_protocol','09_image_score_protocol'),('09_native_restore_summary','10_native_restore_summary'),('12_runtime_memory','13_runtime_memory'),('13_integrity_audit','14_integrity_audit'),('10_evaluation_report','11_evaluation_report'),('11_aupro_exact_report','12_aupro_exact_report'),('14_comparison_with_existing','15_comparison_with_external_baselines')]:e=e.replace(a,b)
(ROOT/'scripts/fastflow_p2_evaluation.py').write_text(e,encoding='utf8')
