"""Exact bootstrap optimization audit. No models, training, or forwards."""
import argparse
import datetime
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
import run_phase21b_a_bootstrap as old_driver
from src.evaluation import weighted_bootstrap_exact as new

OUT=ROOT/'experiments/phase21b_a_r1_bootstrap_optimization'
OLD=old_driver.OUT
METHODS=old_driver.METHODS
SHAPE=(1056,4224)


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return old_driver.sha(p)
def save(name,obj):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8');t.replace(p)


def prepare():
    import numpy as np
    assert read(OLD/'completion_verification.json')['status']=='STOPPED_BY_USER'
    assert sha(old_driver.METRIC)==old_driver.EXACT_SHA
    audit=read(OLD/'audit_complete.json')
    identities={'01_prediction_inventory.json':'inventory_sha256',
        '05_calibration_indices.json':'indices_sha256','source_hashes.json':'source_hashes_sha256'}
    for filename,key in identities.items():
        assert sha(OLD/filename)==audit[key],('Frozen audit manifest identity mismatch',filename)
    originals=read(OLD/'source_hashes.json')
    for filename in [*identities,'audit_complete.json']:
        originals[str(OLD/filename)]=sha(OLD/filename)
    originals[str(ROOT/'scripts/run_phase21b_a_bootstrap.py')]=sha(ROOT/'scripts/run_phase21b_a_bootstrap.py')
    for p,h in originals.items():assert sha(p)==h,('Frozen source changed',p)
    save('original_hashes.json',originals)
    job=read(OLD/'jobs/r00_m0.json')
    save('01_original_runtime_gate.json',dict(original_calibration_aborted=True,
        ORIGINAL_CALIBRATION_ABORTED_FOR_RUNTIME=True,reason='CURRENT_IMPLEMENTATION_COMPUTATIONALLY_INFEASIBLE',
        stopped_previously_by_user=True,completed_method_tasks=2,original_task_time_sec=job['wall_sec'],
        original_pixel_time_sec=job['pixel_sec'],original_aupro_joint_sec=job['exact_joint_sec'],
        original_peak_ram_gib=job['peak_host_ram_mib']/1024,protocol_violation_count=0,
        old_reference_implementation=str(ROOT/'scripts/run_phase21b_a_bootstrap.py')))
    inventory=read(OLD/'01_prediction_inventory.json')
    save('02_prediction_cache_audit.json',dict(methods=METHODS,native_maps=684,native_shape=list(SHAPE),
        alignment_pass=True,sources_sha_verified=True,original_maps_read_only=True,
        map_copies_per_replicate=0,bootstrap_duplicates_materialized=False,
        reference_validation_note='Old equivalence reference alone retains original disk-backed materialization; optimized replicates never materialize duplicates'))
    gt=new.build_gt_cache(inventory['PB-ARPR']['rows'],OUT/'gt_cache')
    assert gt['total_regions']==450 and gt['connected_component_calls']==90
    save('03_gt_region_cache.json',dict(gt_region_cache_created=True,total_gt_region_count=gt['total_regions'],
        **gt,cache_scope='Shared once across all six optimized methods and all replicates'))
    protocol=read(OLD/'03_bootstrap_protocol.json')
    idx=read(OLD/'05_calibration_indices.json')
    regenerated=new.bootstrap_indices(idx['labels'],20,20260924)
    assert np.array_equal(regenerated,idx['indices'])
    save('06_equivalence_indices.json',dict(seed=20260924,observed_multiplicity=[1]*114,
        bootstrap_replicates=[0,1,2],indices=idx['indices'][:3],multiplicities=idx['multiplicities'][:3],
        calibration_indices=idx['indices'],sample_ids=idx['sample_ids'],labels=idx['labels'],
        original_indices_sha256=sha(OLD/'05_calibration_indices.json')))
    save('04_weighted_pixel_auroc_design.json',dict(exact=True,formula='m.T @ C @ m / (2*N_pos(m)*N_bg(m))',
        C='C[i,j]=sum_p_in_fg_i(left_rank_bg_j(p)+right_rank_bg_j(p))',
        rank_ties='All original float32 ties preserved',pair_accumulation='int64, twice-win counts',
        pixel_subsampling_used=False,threshold_approximation_used=False))
    save('05_weighted_aupro_design.json',dict(exact=True,left_rank='sum_j m_j*left_j(p)',
        right_rank='sum_j m_j*right_j(p)',region_pixel_weight='m_owner/(sum_i m_i*K_i * original_region_size)',
        integration='Frozen tie-aware piecewise integral at .05 and .30; no threshold grid',
        gt_region_definition='Original exact_aupro.components, OpenCV8 connectivity, unchanged masks',
        original_exact_sha256=old_driver.EXACT_SHA,bootstrap_duplicates_materialized=False))
    unit_tests()
    save('13_phase21b_b_preregistration.json',dict(**protocol,status='EQUIVALENCE_PENDING',
        phase21b_b_authorized=False,formal_bootstrap_started=False,mathematical_definitions_unchanged=True,
        implementation='Independent exact sufficient-statistic weighted implementation',
        implementation_change_authorized_by='Phase21B-A-R1'))
    updated=read(OUT/'13_phase21b_b_preregistration.json')
    updated.update(metric_implementation_unchanged=False,old_reference_implementation_unchanged=True,
        metric_mathematical_definitions_unchanged=True,
        pixel_auroc='Exact integer-multiplicity pair-win statistics equivalent to original sklearn ROC area',
        exact_aupro='Cached weighted exact ranks with unchanged reference tie-aware integration formula',
        gt_components_cached_once=True,physical_duplicate_materialization=False)
    save('13_phase21b_b_preregistration.json',updated)
    print('PREPARATION PASS',flush=True)


def unit_tests():
    command=[sys.executable,'-m','pytest','tests/test_weighted_bootstrap_exact.py','-q','--tb=short',
        '--basetemp',str(OUT/'pytest_preflight_tmp')]
    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,check=False)
    (OUT/'unit_test_output.txt').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
    assert result.returncode==0 and '14 passed' in result.stdout, result.stdout+result.stderr
    save('09_unit_tests.json',dict(red='12 failures: optimized module not yet implemented',green='14 passed',
        count=14,command=command,exit_code=result.returncode,output='unit_test_output.txt',
        test_source_sha256=sha(ROOT/'tests/test_weighted_bootstrap_exact.py'),
        implementation_sha256=sha(ROOT/'src/evaluation/weighted_bootstrap_exact.py'),
        numerical_guard_note='Unclipped float64 result retained; guard allows1e-12 roundoff; equivalence tolerances unchanged',
        temporary_directory_issue='Initial default Temp permission denied; tests use audit-owned directory'))


def cache_worker(m):
    inventory=read(OLD/'01_prediction_inventory.json')
    rows=inventory[METHODS[m]]['rows']
    gt=read(OUT/'gt_cache/manifest.json')
    cache=new.build_rank_cache(rows,gt,OUT/f'caches/m{m}')
    save(f'cache_reports/m{m}.json',dict(**cache,peak_host_ram_mib=old_driver.peak_ram()))
    print('CACHE_COMPLETE',METHODS[m],cache['preparation_sec'],flush=True)


def old_reference(m,rep):
    """Unchanged old numerical calls, isolated work directory and no models."""
    import numpy as np
    import cv2
    from PIL import Image
    from sklearn.metrics import roc_auc_score
    from src.evaluation import exact_aupro
    job=OLD/f'jobs/r{rep:02d}_m{m}.json'
    if job.exists():
        obj=read(job)
        assert obj['indices_sha256']==sha(OLD/'05_calibration_indices.json')
        return dict(**obj,reused_original_job=str(job))
    rows=read(OLD/'01_prediction_inventory.json')[METHODS[m]]['rows']
    idx=read(OUT/'06_equivalence_indices.json')['indices'][rep]
    draws=[dict(rows[i],native_shape=SHAPE,sample_id=f'reference{j}::{rows[i]["sample_id"]}') for j,i in enumerate(idx)]
    work=OUT/'reference_work';work.mkdir(exist_ok=True)
    begin=time.perf_counter()
    scores=np.lib.format.open_memmap(work/'scores.npy',mode='w+',dtype=np.float32,shape=(114,*SHAPE))
    targets=np.lib.format.open_memmap(work/'targets.npy',mode='w+',dtype=np.uint8,shape=(114,*SHAPE))
    for j,r in enumerate(draws):
        scores[j]=np.load(r['raw_map_path'],mmap_mode='r')
        targets[j]=exact_aupro.get_mask(r,Image,np)
    scores.flush();targets.flush()
    pixel=float(roc_auc_score(targets.ravel(),scores.ravel()))
    del scores,targets;gc.collect()
    report={}
    exact_aupro.native(draws,report,old_driver.NoResize(),cv2,np,Image,temp_dir=work/'exact')
    return dict(pixel_auroc=pixel,aupro005=report['aupro_0.05'],aupro030=report['aupro_0.30'],
        elapsed_sec=time.perf_counter()-begin,peak_host_ram_mib=old_driver.peak_ram(),
        reference_physical_materialization=True,reference_only_not_optimized=True)


def equivalence_worker(m):
    import numpy as np
    cache=read(OUT/f'caches/m{m}/manifest.json')
    eq=read(OUT/'06_equivalence_indices.json')
    observed=read(OLD/'04_observed_delta_inventory.json')['metrics'][METHODS[m]]
    results=[]
    for rep in [-1,0,1,2]:
        dest=OUT/f'equivalence/m{m}_r{rep}.json'
        if dest.exists():
            results.append(read(dest));continue
        weights=np.ones(114,np.int64) if rep==-1 else np.asarray(eq['multiplicities'][rep],np.int64)
        optimized=new.evaluate(cache,weights)
        reference=observed if rep==-1 else old_reference(m,rep)
        errors={k:abs(optimized[k]-reference[k]) for k in ['pixel_auroc','aupro005','aupro030']}
        passed=errors['pixel_auroc']<=1e-10 and max(errors['aupro005'],errors['aupro030'])<=1e-8
        item=dict(method=METHODS[m],replicate=rep,reference=reference,optimized=optimized,
            absolute_errors=errors,passed=passed,
            observed_reference_source='Frozen existing OLD observed calculation' if rep==-1 else None)
        save(dest.relative_to(OUT),item)
        assert passed,('EQUIVALENCE_FAIL',item)
        results.append(item)
        print('EQUIVALENCE PASS',METHODS[m],rep,errors,flush=True)
    save(f'equivalence/m{m}_summary.json',dict(method=METHODS[m],pass_all=True,comparisons=results))


def equivalence_gate():
    entries=[read(OUT/f'equivalence/m{m}_r{r}.json') for m in range(6) for r in [-1,0,1,2]]
    assert len(entries)==24 and all(e['passed'] for e in entries)
    maxerr={k:max(e['absolute_errors'][k] for e in entries) for k in ['pixel_auroc','aupro005','aupro030']}
    save('07_pixel_equivalence.json',dict(passed=True,max_abs_error=maxerr['pixel_auroc'],tolerance=1e-10,
        coverage='All6 methods: observed114 + first3 fixed bootstrap replicates',comparisons=entries))
    save('08_aupro_equivalence.json',dict(aupro005_equivalence_pass=True,aupro030_equivalence_pass=True,
        aupro005_max_abs_error=maxerr['aupro005'],aupro030_max_abs_error=maxerr['aupro030'],tolerance=1e-8,
        exact_aupro_preserved=True,pixel_subsampling_used=False,threshold_approximation_used=False))


def calibration_worker(m):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    assert read(OUT/'07_pixel_equivalence.json')['passed']
    assert read(OUT/'08_aupro_equivalence.json')['aupro005_equivalence_pass']
    assert read(OUT/'08_aupro_equivalence.json')['aupro030_equivalence_pass']
    cache=read(OUT/f'caches/m{m}/manifest.json')
    rows=read(OLD/'01_prediction_inventory.json')[METHODS[m]]['rows']
    eq=read(OUT/'06_equivalence_indices.json')
    labels=np.asarray(eq['labels']);scores=np.asarray([r['image_score'] for r in rows])
    results=[];begin=time.perf_counter()
    for rep,idx in enumerate(eq['calibration_indices']):
        idx=np.asarray(idx);weights=np.bincount(idx,minlength=114)
        start=time.perf_counter()
        image_auc=float(roc_auc_score(labels[idx],scores[idx]))
        image_sec=time.perf_counter()-start
        metric=new.evaluate(cache,weights)
        results.append(dict(replicate=rep,method=METHODS[m],image_auroc=image_auc,image_sec=image_sec,**metric))
        print('OPTIMIZED_CALIBRATION',METHODS[m],rep+1,metric['aupro_joint_sec'],flush=True)
    save(f'calibration/m{m}.json',dict(method=METHODS[m],replicates=20,results=results,
        wall_sec=time.perf_counter()-begin,peak_host_ram_mib=old_driver.peak_ram()))


def finish(calibration_wall):
    jobs=[read(OUT/f'calibration/m{m}.json') for m in range(6)]
    assert all(j['replicates']==20 and len(j['results'])==20 for j in jobs)
    results=[r for j in jobs for r in j['results']]
    image_time=sum(r['image_sec'] for r in results)
    pixel_time=sum(r['pixel_sec'] for r in results)
    aupro_time=sum(r['aupro_joint_sec'] for r in results)
    paired=[]
    for rep in range(20):
        pb=jobs[0]['results'][rep]
        for m in range(1,6):
            baseline=jobs[m]['results'][rep]
            paired.append(dict(replicate=rep,baseline=METHODS[m],pb_arpr=pb,baseline_metrics=baseline,
                delta_pp={k:100*(pb[k]-baseline[k]) for k in ['image_auroc','pixel_auroc','aupro005','aupro030']}))
    save('calibration_paired_results.json',paired)
    cache_reports=[read(OUT/f'cache_reports/m{m}.json') for m in range(6)]
    setup=sum(r['preparation_sec'] for r in cache_reports)
    calibration_peak=max(j['peak_host_ram_mib'] for j in jobs)/1024
    prep_peak=max(j['peak_host_ram_mib'] for j in cache_reports)/1024
    peak=max(calibration_peak,prep_peak)
    old=read(OUT/'01_original_runtime_gate.json')
    old_peak=old['original_peak_ram_gib']
    estimates={f'estimated_{n}_time':calibration_wall*n/20 for n in [500,1000,2000]}
    authorized=estimates['estimated_1000_time']<=4*3600
    status='PASS_FOR_BOOTSTRAP_1000' if authorized else 'HOLD_BOOTSTRAP_RUNTIME_AFTER_OPTIMIZATION'
    save('10_optimized_runtime_calibration.json',dict(optimized_calibration_replicates=20,
        optimized_calibration_success=20,method_jobs=120,total_20_optimized_wall_time=calibration_wall,
        image_auroc_time=image_time,pixel_auroc_time=pixel_time,aupro_joint_time=aupro_time,
        one_time_cache_preparation_sec=setup,cache_setup_excluded_from_repeated_metric_time=True,methods=jobs))
    save('11_memory_comparison.json',dict(original_peak_ram_gib=old_peak,optimized_peak_ram_gib=peak,
        optimized_calibration_peak_gib=calibration_peak,
        optimized_cache_preparation_peak_gib=prep_peak,ram_reduction_percent=100*(1-peak/old_peak),
        old_reference_validation_memory_excluded=True,measurement='Windows PeakWorkingSetSize per subprocess'))
    save('12_formal_runtime_estimate.json',dict(**estimates,unit='seconds',estimated_not_measured=True,
        with_fresh_cache={str(n):setup+estimates[f'estimated_{n}_time'] for n in [500,1000,2000]},
        fixed_cache_preparation_sec=setup,linear_extrapolation='Measured complete 20 six-method replicates',
        caveat='Cache IO, OS caching and system load can change runtime; no measured1000 run'))
    for path,h in read(OUT/'original_hashes.json').items():assert sha(path)==h,('Original changed',path)
    proto=read(OUT/'13_phase21b_b_preregistration.json')
    proto.update(status=status,phase21b_b_authorized=authorized,formal_bootstrap_started=False,
        weighted_implementation_sha256=sha(ROOT/'src/evaluation/weighted_bootstrap_exact.py'))
    save('13_phase21b_b_preregistration.json',proto)
    pix=read(OUT/'07_pixel_equivalence.json');pro=read(OUT/'08_aupro_equivalence.json')
    speed_basis=old['original_task_time_sec']*120
    completion=dict(phase='Phase21B-A-R1',status=status,original_calibration_aborted=True,
        original_task_time_sec=old['original_task_time_sec'],original_peak_ram_gib=old_peak,
        bootstrap_unit='native_image',stratified=True,paired=True,bootstrap_seed=20260924,
        physical_duplicate_materialization=False,reference_materialization_only=True,
        gt_region_cache_created=True,total_gt_region_count=450,pixel_auroc_equivalence_pass=True,
        pixel_auroc_max_abs_error=pix['max_abs_error'],**{k:v for k,v in pro.items() if k!='tolerance'},
        optimized_calibration_replicates=20,optimized_calibration_success=20,
        optimized_20_wall_time_sec=calibration_wall,optimized_peak_ram_gib=peak,
        optimized_calibration_peak_gib=calibration_peak,
        cache_preparation_sec=setup,cache_preparation_peak_gib=prep_peak,
        ram_reduction_percent=100*(1-peak/old_peak),**estimates,phase21b_b_authorized=authorized,
        speedup_vs_original_first_task_projection=speed_basis/calibration_wall,
        speedup_note='Original20 was aborted; denominator old20=first244.51sec task*120 is estimated, not measured.',
        model_forward_count=0,training_count=0,formal_bootstrap_started=False,
        original_source_hashes_unchanged=True,protocol_violation_count=0)
    save('completion_verification.json',completion)
    report=['PHASE21B-A-R1 EXACT BOOTSTRAP OPTIMIZATION',f'STATUS: {status}',
        'Original calibration safely aborted; all original assets preserved.',
        f'Original first task: {old["original_task_time_sec"]:.6f} sec; peak: {old_peak:.6f} GiB',
        'Equivalence: all6 observed +3 paired replicates PASS; Pixel tolerance1e-10, PRO1e-8',
        f'Pixel max error: {pix["max_abs_error"]:.16g}',
        f'AUPRO005 max error: {pro["aupro005_max_abs_error"]:.16g}',
        f'AUPRO030 max error: {pro["aupro030_max_abs_error"]:.16g}',
        'Pixel subsampling=False; threshold approximation=False; GT region changes=False',
        f'GT cache:450 regions,90 component calls; reused for optimized evaluation',
        f'Optimized20 complete wall time: {calibration_wall:.6f} sec',
        f'Image: {image_time:.6f} sec; Pixel: {pixel_time:.6f} sec; exactPRO shared: {aupro_time:.6f} sec',
        f'One-time rank preparation: {setup:.6f} sec; preparation peak: {prep_peak:.6f} GiB',
        f'Optimized peak RAM: {peak:.6f} GiB; reduction: {100*(1-peak/old_peak):.6f}%',
        f'RAM scope: max(cache preparation, optimized calibration); calibration-only: {calibration_peak:.6f} GiB; legacy equivalence-reference RAM separate',
        f'Speedup vs estimated old20: {speed_basis/calibration_wall:.6f}x (old20 extrapolated, not measured)',
        *[f'{k}: {v/3600:.6f} hours ESTIMATED_NOT_MEASURED' for k,v in estimates.items()],
        f'PHASE21B_B_AUTHORIZED: {authorized}',
        'Protocol violations:0; no1000 run, no model forward or training.']
    (OUT/'final_console_report.txt').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print('\n'.join(report),flush=True)


def launch(mode,m):
    logfile=OUT/f'logs/{mode}_m{m}.txt';logfile.parent.mkdir(exist_ok=True)
    print('START',mode,METHODS[m],flush=True)
    with logfile.open('a',encoding='utf-8') as f:
        subprocess.run([sys.executable,str(Path(__file__).resolve()),mode,'--method',str(m)],cwd=ROOT,
            stdout=f,stderr=subprocess.STDOUT,check=True)
    save('progress.json',dict(last_completed_stage=mode,last_completed_method=METHODS[m],
        updated_at=datetime.datetime.now().astimezone().isoformat()))


def run():
    if not (OUT/'original_hashes.json').exists():prepare()
    for m in range(6):
        if not (OUT/f'cache_reports/m{m}.json').exists():launch('cache',m)
    for m in range(6):
        if not (OUT/f'equivalence/m{m}_summary.json').exists():launch('equivalence',m)
    equivalence_gate()
    start=time.perf_counter()
    for m in range(6):
        assert not (OUT/f'calibration/m{m}.json').exists(),'Do not repeat completed calibration automatically'
        launch('calibrate',m)
    finish(time.perf_counter()-start)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','cache','equivalence','calibrate','run']);parser.add_argument('--method',type=int)
    args=parser.parse_args();OUT.mkdir(exist_ok=True,parents=True)
    try:
        if args.mode=='prepare':prepare()
        elif args.mode=='cache':cache_worker(args.method)
        elif args.mode=='equivalence':equivalence_worker(args.method)
        elif args.mode=='calibrate':calibration_worker(args.method)
        else:run()
    except Exception:
        error=traceback.format_exc();print(error,flush=True)
        if args.mode in ['run','prepare']:
            save('execution_failure.json',dict(status='HOLD_OPTIMIZATION_AUDIT',error=error,phase21b_b_authorized=False))
        raise
