"""Normal-training-only geometry recheck; no model or test-image access."""
import ast,math,json,time,statistics
from pathlib import Path
from PIL import Image
from run_audit import ROOT,OUT,sha,read,rows,save,csvsave,resources,now
T=time.perf_counter();sources={}
def track(rel):
    p=ROOT/rel;sources[str(p)]=sha(p);return p
def j(rel):return read(track(rel))
def c(rel):return rows(track(rel))
def function(rel,name,env):
    p=track(rel);tree=ast.parse(p.read_text(encoding='utf-8-sig'));node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),env);return env[name]
def main():
    dims=[];summaries=[]
    for dataset,manifest,expected in [('sheet_metal',None,137),('KSDD2','experiments/phase15a_ksdd2_audit/split_manifest.csv',1876),('Severstal','experiments/phase16a_severstal_audit/split_manifest.csv',4131)]:
        if manifest:
            full=c(manifest);selected=[r for r in full if r['split']=='train_normal'];assert len(selected)==expected
        else:selected=[dict(image_path=str(p),label='0') for p in sorted((ROOT/'datasets/MVTec_AD_2/sheet_metal/train/good').glob('*.png'))];assert len(selected)==expected
        rr=[]
        for r in selected:
            assert int(r['label'])==0;p=Path(r['image_path']);assert p.is_relative_to(ROOT/'datasets')
            with Image.open(p) as im:w,h=im.size
            digest=sha(p)
            if manifest:assert digest==r['image_sha'] and h==int(r['height']) and w==int(r['width'])
            dims.append(dict(dataset=dataset,path=str(p),height=h,width=w,aspect_hw=h/w,sha256=digest,split='train_normal'));rr.append(h/w)
        ratio=statistics.median(rr)
        raw=function('src/evaluation/ksdd2_external_protocol.py','select_rect',{'math':math})(ratio)
        summaries.append(dict(dataset=dataset,n_normal_train=len(selected),median_h_over_w=ratio,raw_selector_hw=list(raw),status='VERIFIED',historical_hw=[256,1024] if dataset=='sheet_metal' else [864,320] if dataset=='KSDD2' else [256,1152]))
        print('NORMAL_GEOMETRY',summaries[-1],flush=True)
    r1='experiments/phase16a_r1_severstal_geometry_revision/'
    g=j(r1+'geometry_protocol_r1.json');binding=j(r1+'geometry_binding.json')
    assert sha(ROOT/r1/'geometry_protocol_r1.json')==binding['GEOMETRY_R1_SHA256']
    inventory=c(r1+'geometry_candidate_inventory.csv');assert sha(ROOT/r1/'geometry_candidate_inventory.csv')==g['inventory_sha']
    assert len(inventory)==g['candidate_count']
    legal=[(int(r['H']),int(r['W'])) for r in inventory if r['feasible']=='True']
    env={'math':math,'frozen_config':lambda:dict(target_pixels=262144)}
    fallback=function('src/common/architecture_constrained_geometry.py','choose',env)(legal,summaries[-1]['median_h_over_w'])
    assert list(fallback)==g['selected_hw']==[256,1152]
    summaries[-1]['fallback_hw']=list(fallback);summaries[-1]['fallback_evidence']='Re-rank hash-verified historical full-forward inventory; no new architecture probes'
    execution=j(r1+'execution_binding.json');codechecks=[]
    for rel,h in execution['code_sha256'].items():
        actual=sha(track(rel));codechecks.append(dict(path=str(ROOT/rel),expected=h,actual=actual,match=h==actual))
    assert all(r['match'] for r in codechecks)
    original=j('experiments/phase16a_severstal_audit/geometry_protocol.json')
    assert sha(ROOT/'src/evaluation/ksdd2_external_protocol.py')==original['source_file_sha']
    split=j('experiments/phase16a_severstal_audit/split_protocol.json')
    # Reproduce canonical split digests without reading masks or held-out images.
    severstal=c('experiments/phase16a_severstal_audit/split_manifest.csv')
    ss=sha(track('experiments/phase16a_severstal_audit/split_manifest.csv'))
    canonical=__import__('hashlib').sha256(json.dumps([{k:str(r[k]) for k in ('id','split','image_sha','mask_sha','signature')} for r in sorted(severstal,key=lambda r:r['id'])],sort_keys=True).encode()).hexdigest()
    assert canonical==split['canonical_split_sha256']==g['split_sha']
    ks=c('experiments/phase15a_ksdd2_audit/split_manifest.csv');kprotocol=j('experiments/phase15a_ksdd2_audit/external_protocol.json')
    canonical_k=__import__('hashlib').sha256(json.dumps([{k:r[k] for k in ('id','split','image_sha')} for r in sorted(ks,key=lambda r:r['id'])],sort_keys=True).encode()).hexdigest()
    assert canonical_k==kprotocol['split_sha']
    summaries[1]['historical_all_data_ratio']=statistics.median(float(r['height'])/float(r['width']) for r in ks)
    trainrecords=[]
    for ds in ('KSDD2','Severstal'):
        for seed in (range(42,47) if ds=='KSDD2' else [42]):
            for geo in ('STD','RECT'):
                if ds=='KSDD2':
                    sub=('std_256' if geo=='STD' else 'rect_864x320')
                    d=f'experiments/phase15c_ksdd2_70k_formal_dev/{sub}_seed42' if seed==42 else f'experiments/phase15d_ksdd2_5seed_repeats/seed{seed}/{sub}'
                else:d='experiments/phase16c_severstal_70k_formal_dev/'+('std_256_seed42' if geo=='STD' else 'rect_256x1152_seed42')
                tr=j(d+'/training_report.json');cfg=j(d+'/config.json');cp=Path(tr['checkpoint']);actual=sha(cp)
                assert actual==tr['checkpoint_sha256']
                log=track(d+'/loss_history.csv');lr=rows(log)
                trainrecords.append(dict(dataset=ds,seed=seed,geometry=geo,input_hw=str(tr['input_hw']),config_path=str(ROOT/d/'config.json'),config_sha256=sha(ROOT/d/'config.json'),report_path=str(ROOT/d/'training_report.json'),checkpoint_path=str(cp),checkpoint_sha256=actual,log_path=str(log),log_sha256=sha(log),log_rows=len(lr),last_log_row=json.dumps(lr[-1]),report_steps=tr.get('global_step',tr.get('steps')),geometry_sha=tr.get('geometry_sha',''),status='VERIFIED'))
                if ds=='Severstal':assert tr['geometry_sha']==binding['GEOMETRY_R1_SHA256'] and tr['split_sha']==canonical
    for rel in ['src/evaluation/rectangular_efficientad_calibration.py','scripts/train_rectangular_efficientad_256x1024.py','scripts/phase24b_train_rect.py','src/evaluation/ksdd2_70k_protocol.py','src/training/ksdd2_repeat_training.py','src/evaluation/severstal_protocol.py','src/evaluation/severstal_70k_protocol.py','src/common/pbarpr_geometry.py','experiments/phase15a_ksdd2_audit/dataset_audit.json','experiments/phase16a_severstal_audit/dataset_audit.json','experiments/phase16a_r1_severstal_geometry_revision/regression_report.json','PB_ARPR_Paper_Materials/09_Paper_Outline/contributions.md','PB_ARPR_Paper_Materials/09_Paper_Outline/discussion_limitations.md','PB_ARPR_Paper_Materials/03_Tables_Main/Table_03_Geometry_Ablation.md','PB_ARPR_Paper_Materials/03_Tables_Main/Table_04_Three_Dataset_Generalization.md']:
        track(rel)
    csvsave('normal_training_dimensions.csv',dims);csvsave('historical_training_manifest.csv',trainrecords)
    csvsave('geometry_source_manifest.csv',[dict(path=p,sha256=h) for p,h in sources.items()])
    save('geometry_recheck.json',dict(status='PASS',created=now(),datasets=summaries,code_bindings=codechecks,severstal_inventory=dict(candidates=len(inventory),feasible=len(legal),selected=list(fallback),geometry_sha=binding['GEOMETRY_R1_SHA256']),canonical_split_hashes=dict(KSDD2=canonical_k,Severstal=canonical),seconds=time.perf_counter()-T,resources=resources(),limits='Historical statements require source plus bound records; normal-only recomputation is retrospective, not proof of historical normal-only execution'))
    print('GEOMETRY_RECHECK_PASS',json.dumps(resources()),flush=True)
if __name__=='__main__':main()
