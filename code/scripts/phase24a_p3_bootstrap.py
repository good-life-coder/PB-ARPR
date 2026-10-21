"""Phase24A-P3 paired stratified bootstrap from frozen Phase24A-P2 predictions."""
import csv,hashlib,json,math,os,sys,time,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[1]
P=R/'experiments/phase24a_p2_patchcore_geometry_formal'
O=R/'experiments/phase24a_p3_patchcore_geometry_bootstrap'
D=R/'datasets/MVTec_AD_2/sheet_metal'
sys.path.insert(0,str(R))
import numpy as np
from src.evaluation import exact_aupro as old
from src.evaluation import weighted_bootstrap_exact as weighted
SEED=20260926
METRICS={'image_auroc':'image_auroc','pixel_auroc':'pixel_auroc','aupro005':'exact_aupro005','aupro030':'exact_aupro030'}
FROZEN={'square':dict(image_auroc=63.425926,pixel_auroc=86.206080,exact_aupro005=13.495947,exact_aupro030=32.461140),
        'rect':dict(image_auroc=75.324074,pixel_auroc=91.379412,exact_aupro005=27.289232,exact_aupro030=58.340818),
        'delta':dict(image_auroc=11.898148,pixel_auroc=5.173332,exact_aupro005=13.793285,exact_aupro030=25.879678)}
accessed=[];blocked=[]
def audit(event,args):
    if event=='socket.connect':raise PermissionError('offline')
    if event not in ('open','os.listdir','os.scandir') or not args or not isinstance(args[0],(str,bytes)):return
    p=Path(os.fsdecode(args[0])).absolute()
    if p.is_relative_to(R/'datasets'):
        if not p.is_relative_to(D/'test_public/ground_truth/bad'):
            blocked.append(str(p));raise PermissionError(str(p))
        if event=='open':accessed.append(str(p))
sys.addaudithook(audit)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(n,obj):
    path=O/n;path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(str(path))
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
def gate():
    c=read(P/'completion_verification.json')
    report=(P/'final_console_report.txt').read_text(encoding='utf8')
    integrity=read(P/'12_prediction_integrity.json')
    banks=read(P/'02_bank_verification.json')
    assert c['status']=='PASS_PATCHCORE_GEOMETRY_FORMAL_COMPLETE' and c['status'] in report
    assert c['square_prediction_count']==c['rect_prediction_count']==114
    assert c['prediction_alignment_pass'] is True and integrity['prediction_alignment_pass'] is True
    assert all(integrity[k]==0 for k in ['missing_count','duplicate_count','nan_image_scores','inf_image_scores','nan_anomaly_pixels','inf_anomaly_pixels'])
    assert c['square_bank_sha_match'] and c['rect_bank_sha_match'] and all(x['sha_match'] for x in banks.values())
    assert c['test_private_files_accessed']==c['test_private_mixed_files_accessed']==c['protocol_violation_count']==0
    for tag in ['square','rect']:
        for key in METRICS.values():
            assert abs(c[tag+'_'+key]*100-FROZEN[tag][key])<.00000051,(tag,key)
    for key in METRICS.values():
        assert abs(c['delta_'+key+'_pp']-FROZEN['delta'][key])<.00000051,key
    rows={}
    for tag,dir in [('square','04_square512_predictions'),('rect','05_rect_predictions')]:
        rows[tag]=read(P/dir/'manifest.json')
        assert len(rows[tag])==len({r['sample_id'] for r in rows[tag]})==114
        assert len(list((P/dir).glob('*_native.npy')))==len(list((P/dir).glob('*_model.npy')))==114
        for row in rows[tag]:
            assert Path(row['raw_map_path']).is_relative_to(P/dir)
            assert Path(row['raw_map_path']).is_file() and sha(row['raw_map_path'])==row['native_map_sha256']
            assert math.isfinite(row['official_image_score'])
    assert [r['sample_id'] for r in rows['square']]==[r['sample_id'] for r in rows['rect']]
    assert [r['gt_label'] for r in rows['square']]==[r['gt_label'] for r in rows['rect']]
    assert sum(r['gt_label'] for r in rows['square'])==90
    assert sha(R/'src/evaluation/exact_aupro.py')=='3e6ebc80d400419cf3aaf00950e954ea165df2419c6ea895f4296013c67e11a1'
    return c,rows
def normalized(rows):
    return [dict(r,native_shape=[1056,4224],raw_map_path=r['raw_map_path'],image_score=r['official_image_score']) for r in rows]
def setup():
    c,rows=gate()
    if not O.exists():O.mkdir()
    if not (O/'01_previous_phase_gate.json').exists():
        save('01_previous_phase_gate.json',dict(status='PASS',previous_status=c['status'],previous_report_sha256=sha(P/'final_console_report.txt'),previous_completion_sha256=sha(P/'completion_verification.json'),prediction_count_per_geometry=114,prediction_alignment_pass=True,bank_sha_match=True,nonfinite_count=0,private_access_count=0,protocol_violation_count=0,frozen_metrics_percent=FROZEN))
    labels=np.asarray([r['gt_label'] for r in rows['square']],dtype=np.uint8)
    indices=weighted.bootstrap_indices(labels,1000,SEED)
    assert indices.shape==(1000,114) and all((labels[x[:24]]==0).all() and (labels[x[24:]]==1).all() for x in indices)
    if not (O/'bootstrap_indices.npz').exists():np.savez(O/'bootstrap_indices.npz',indices=indices,labels=labels,sample_ids=np.asarray([r['sample_id'] for r in rows['square']]))
    else:
        with np.load(O/'bootstrap_indices.npz') as a:assert np.array_equal(a['indices'],indices)
    if not (O/'02_bootstrap_protocol.json').exists():
        save('02_bootstrap_protocol.json',dict(unit='native image',paired=True,stratified=True,replacement=True,seed=SEED,replicates=1000,normal_count=24,anomaly_count=90,normal_draws=24,anomaly_draws=90,shared_indices=True,native_shape=[1056,4224],pixel_subsampling=False,threshold_approximation=False,bootstrap_metric='Rect minus Square',confidence_interval='percentile 2.5% and 97.5%',success_threshold=None,weighted_module_sha256=sha(R/'src/evaluation/weighted_bootstrap_exact.py'),exact_module_sha256=sha(R/'src/evaluation/exact_aupro.py')))
    nr=normalized(rows['square']);gt=weighted.build_gt_cache(nr,O/'gt_cache')
    assert gt['total_regions']==450
    caches={}
    for tag in ['square','rect']:
        caches[tag]=weighted.build_rank_cache(normalized(rows[tag]),gt,O/('rank_'+tag))
        print('RANK_CACHE',tag,caches[tag]['positive_count'],flush=True)
    for tag in caches:
        m=weighted.evaluate(caches[tag],np.ones(114,dtype=np.int64))
        assert abs(m['pixel_auroc']-c[tag+'_pixel_auroc'])<1e-10
        assert abs(m['aupro005']-c[tag+'_exact_aupro005'])<1e-10
        assert abs(m['aupro030']-c[tag+'_exact_aupro030'])<1e-10
    return c,rows,indices,caches
def equivalence(rows,indices,caches):
    import cv2
    from PIL import Image
    rng=np.random.default_rng(SEED+1)
    selected=sorted(rng.choice(1000,20,replace=False).tolist())
    records=[]
    for rep in selected:
        idx=indices[rep];mult=np.bincount(idx,minlength=114).astype(np.int64)
        for tag in ['square','rect']:
            dest=O/'equivalence'/('%04d_%s.json'%(rep,tag))
            if dest.exists():item=read(dest);records.append(item);continue
            optimized=weighted.evaluate(caches[tag],mult)
            draws=[dict(rows[tag][i],native_shape=(1056,4224),sample_id='%04d:%s'%(j,rows[tag][i]['sample_id'])) for j,i in enumerate(idx)]
            ref={}
            old.native(draws,ref,None,cv2,np,Image,temp_dir=O/'explicit_temp')
            errors=dict(aupro005=abs(optimized['aupro005']-ref['aupro_0.05']),aupro030=abs(optimized['aupro030']-ref['aupro_0.30']))
            item=dict(replicate_id=rep,geometry=tag,exact_reference={'aupro005':ref['aupro_0.05'],'aupro030':ref['aupro_0.30']},weighted={'aupro005':optimized['aupro005'],'aupro030':optimized['aupro030']},absolute_error=errors,pass_gate=max(errors.values())<1e-10)
            save('equivalence/'+dest.name,item)
            records.append(item)
            print('EQUIVALENCE',rep,tag,errors,flush=True)
            if not item['pass_gate']:raise RuntimeError('FAIL_BOOTSTRAP_EQUIVALENCE_GATE')
    max_error=max(max(x['absolute_error'].values()) for x in records)
    assert len(records)==40 and len({x['replicate_id'] for x in records})==20 and max_error<1e-10
    if not (O/'03_equivalence_validation.json').exists():
        save('03_equivalence_validation.json',dict(status='PASS',random_replicate_count=20,geometry_comparisons=40,replicate_ids=selected,max_absolute_error=max_error,tolerance_strict_less_than=1e-10,exact_reference='src/evaluation/exact_aupro.py native() with explicit repeated image rows',weighted_source='src/evaluation/weighted_bootstrap_exact.py',records=records))
    return max_error
def formal(c,rows,indices,caches):
    from sklearn.metrics import roc_auc_score
    out=O/'bootstrap_deltas.csv'
    fields=['replicate_id','delta_image_auroc_pp','delta_pixel_auroc_pp','delta_aupro005_pp','delta_aupro030_pp']
    prior=[]
    if out.exists():
        with out.open(newline='',encoding='utf8') as f:prior=list(csv.DictReader(f))
        assert all(int(x['replicate_id'])==i for i,x in enumerate(prior))
    else:
        with out.open('x',newline='',encoding='utf8') as f:csv.DictWriter(f,fieldnames=fields).writeheader()
    labels=np.asarray([r['gt_label'] for r in rows['square']])
    scores={tag:np.asarray([r['official_image_score'] for r in rows[tag]]) for tag in ['square','rect']}
    with out.open('a',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        for rep in range(len(prior),1000):
            idx=indices[rep];mult=np.bincount(idx,minlength=114).astype(np.int64)
            result={}
            for tag in ['square','rect']:
                z=weighted.evaluate(caches[tag],mult)
                result[tag]=dict(image_auroc=float(roc_auc_score(labels[idx],scores[tag][idx])),pixel_auroc=z['pixel_auroc'],aupro005=z['aupro005'],aupro030=z['aupro030'])
            row=dict(replicate_id=rep,**{'delta_'+k+'_pp':100*(result['rect'][k]-result['square'][k]) for k in ['image_auroc','pixel_auroc','aupro005','aupro030']})
            writer.writerow(row);f.flush()
            if (rep+1)%25==0:print('BOOTSTRAP',rep+1,'/1000',flush=True)
    with out.open(newline='',encoding='utf8') as f:data=list(csv.DictReader(f))
    assert len(data)==1000 and [int(x['replicate_id']) for x in data]==list(range(1000))
    summary={}
    for metric,key in [('image_auroc','image_auroc'),('pixel_auroc','pixel_auroc'),('aupro005','exact_aupro005'),('aupro030','exact_aupro030')]:
        a=np.asarray([float(x['delta_'+metric+'_pp']) for x in data]);assert np.isfinite(a).all()
        lo,hi=np.quantile(a,[.025,.975])
        summary[metric]=dict(observed_delta_pp=c['delta_'+key+'_pp'],bootstrap_mean_delta_pp=float(np.mean(a)),bootstrap_std_pp=float(np.std(a)),ci95_low_pp=float(lo),ci95_high_pp=float(hi),ci_crosses_zero=bool(lo<=0<=hi))
    save('04_bootstrap_summary.json',summary)
    save('05_ci_zero_crossing.json',{k:v['ci_crosses_zero'] for k,v in summary.items()})
    assert not blocked and not any('test_private' in p for p in accessed)
    v=dict(phase='Phase24A-P3',status='PASS_PATCHCORE_GEOMETRY_BOOTSTRAP_COMPLETE',bootstrap_seed=SEED,bootstrap_replicates=1000,normal_count=24,anomaly_count=90,prediction_rerun_count=0,exact_equivalence_pass=True,test_private_files_accessed=0,test_private_mixed_files_accessed=0,protocol_violation_count=0)
    for k,row in summary.items():
        v['observed_delta_'+k+'_pp']=row['observed_delta_pp']
        for suffix,col in [('bootstrap_mean_delta_pp','bootstrap_mean_delta_pp'),('bootstrap_std_pp','bootstrap_std_pp'),('ci95_low_pp','ci95_low_pp'),('ci95_high_pp','ci95_high_pp'),('ci_crosses_zero','ci_crosses_zero')]:
            v[k+'_'+suffix]=row[col]
    save('completion_verification.json',v)
    lines=['PHASE24A-P3 PATCHCORE PAIRED STRATIFIED BOOTSTRAP','STATUS = '+v['status']]
    for name,key in [('Image AUROC','image_auroc'),('Pixel AUROC','pixel_auroc'),('Exact AUPRO@0.05','aupro005'),('Exact AUPRO@0.30','aupro030')]:
        x=summary[key]
        lines.extend([name,'observed delta = %+.6f pp'%x['observed_delta_pp'],'bootstrap mean ± std = %+.6f ± %.6f pp'%(x['bootstrap_mean_delta_pp'],x['bootstrap_std_pp']),'95%% CI = [%+.6f, %+.6f] pp'%(x['ci95_low_pp'],x['ci95_high_pp']),'crosses zero = '+str(x['ci_crosses_zero'])])
    lines+=['bootstrap = 1000/1000','exact equivalence gate = PASS','prediction rerun count = 0','private/private_mixed accesses = 0/0','protocol violations = 0','final_console_report.txt = '+str(O/'final_console_report.txt'),'completion_verification.json = '+str(O/'completion_verification.json')]
    (O/'final_console_report.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('\n'.join(lines),flush=True)
def main():
    if (O/'completion_verification.json').exists():raise RuntimeError('Already complete; no repeat')
    c,rows,indices,caches=setup()
    equivalence(rows,indices,caches)
    formal(c,rows,indices,caches)
if __name__=='__main__':
    try:main()
    except Exception:traceback.print_exc();raise
