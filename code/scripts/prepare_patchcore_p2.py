"""Derive the authorized full-bank runner from the verified R3 orchestration only."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'scripts/smoke_patchcore_r3.py').read_text(encoding='utf8')
s=s[:s.index("        stage='validation';")]
s=s.replace("OUT=ROOT/'experiments/phase20b_p1_r3_patchcore_smoke_resume'", "OUT=ROOT/'experiments/phase20b_p2_patchcore_formal'\nTEST_UNLOCKED=False")
s=s.replace("('train','validation')", "(('train','validation','test_public') if TEST_UNLOCKED else ('train','validation'))")
s=s.replace(" or any(x in ('ground_truth','bad') for x in rel.parts)", "")
s=s.replace("    OUT.mkdir", "    global TEST_UNLOCKED\n    OUT.mkdir",1)
s=s.replace("phase='Phase20B-P1-R3'", "phase='Phase20B-P2'")
s=s.replace("        pa=ROOT/", "        r3=ROOT/'experiments/phase20b_p1_r3_patchcore_smoke_resume'\n        prior=json.loads((r3/'completion_verification.json').read_text(encoding='utf8'))\n        (r3/'final_console_report.txt').read_text(encoding='utf8')\n        assert prior['status']=='PASS_FOR_PATCHCORE_FORMAL' and prior['phase20b_p2_patchcore_formal_authorized'], 'FAIL_PREVIOUS_GATE'\n        assert prior['test_public_files_accessed']==prior['test_mask_files_accessed']==0\n        pa=ROOT/",1)
s=s.replace("cache=OUT/'weights/hub'", "cache=r3/'weights/hub'")
s=s.replace("        (repo/'refs/main').write_text(commit)", "        assert (repo/'refs/main').read_text()==commit")
s=s.replace("train=select_subset(trains,16); val=select_subset(vals,4)", "train=select_subset(trains,137); val=[]")
s=s.replace("selection='lexicographic first16 train and first4 validation'", "selection='all137 normal training sources; no calibration fitting'")
s=s.replace("train_smoke_count=16,validation_smoke_count=4", "train_count=137,validation_count=19")
for a,b in [('01_previous_gates.json','01_preflight.json'),('02_weight_reverification.json','02_weight_verification.json'),('04_patchcore_config.json','03_patchcore_config.json'),('03_smoke_manifest.json','04_train_manifest.json'),('05_memory_bank_audit.json','05_memory_bank_summary.json')]: s=s.replace(a,b)
s+='''        c.update(model_init_sec=c['model_initialization_time_sec'],feature_extraction_sec=ft,coreset_build_sec=ct,total_build_sec=totalfit,
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
        (OUT/'final_console_report.txt').write_text('PHASE20B-P2 PATCHCORE FORMAL EXTERNAL BASELINE\\n'+json.dumps(c,indent=2,ensure_ascii=False)+'\\nMethod-level comparison; no configuration search; FastFlow not started.\\n',encoding='utf8')
        print(json.dumps(c,indent=2,ensure_ascii=False),flush=True)
    return 0 if c['status']=='PASS' else 1
if __name__=='__main__': sys.exit(main())
'''
(ROOT/'scripts/run_patchcore_p2.py').write_text(s,encoding='utf8')
