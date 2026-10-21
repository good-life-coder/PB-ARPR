"""Formal bootstrap orchestration; frozen R1 numerical implementation unchanged."""
import argparse
import csv
import datetime
import json
import sys
import time
import traceback
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
import run_phase21b_a_r1 as r1
from src.evaluation import weighted_bootstrap_exact as exact
from sklearn.metrics import roc_auc_score

OUT = ROOT/'experiments/phase21b_b_bootstrap_1000'
METHODS = r1.METHODS
METRICS = ['image_auroc', 'pixel_auroc', 'aupro005', 'aupro030']
SEED = 20260924
B = 1000
read = r1.read
sha = r1.sha

def save(name, obj):
    path = OUT/name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def summarize(delta, observed):
    assert np.isfinite(delta).all()
    low, high = np.percentile(delta, [2.5, 97.5], method='linear')
    return dict(observed_delta_pp=100*observed,
        bootstrap_mean_delta_pp=100*float(np.mean(delta)),
        bootstrap_median_delta_pp=100*float(np.median(delta)),
        ci_low_pp=100*float(low), ci_high_pp=100*float(high),
        fraction_delta_gt_zero=float(np.mean(delta > 0)))

def validate_job(job, replicate, identity):
    assert job['replicate'] == replicate
    assert job['indices_sha256'] == identity

def prepare():
    begin = time.perf_counter()
    gate = read(r1.OUT/'completion_verification.json')
    assert gate['status']=='PASS_FOR_BOOTSTRAP_1000' and gate['phase21b_b_authorized']
    assert gate['exact_aupro_preserved'] and gate['total_gt_region_count']==450
    limits = {'pixel_auroc':2.1094237467877974e-15,
              'aupro005':1.6653345369377348e-16, 'aupro030':3.3306690738754696e-16}
    for key, limit in limits.items():
        assert gate[key+'_equivalence_pass'] and gate[key+'_max_abs_error']<=limit
    assert sha(ROOT/'src/evaluation/weighted_bootstrap_exact.py') == read(r1.OUT/'09_unit_tests.json')['implementation_sha256']
    inventory = read(r1.OLD/'01_prediction_inventory.json')
    eq = read(r1.OUT/'06_equivalence_indices.json')
    labels = np.asarray(eq['labels'])
    assert len(labels)==114 and sum(labels==0)==24 and sum(labels==1)==90
    indices = exact.bootstrap_indices(labels, B, SEED)
    assert np.array_equal(indices[:20], eq['calibration_indices'])
    weights = np.array([np.bincount(i, minlength=114) for i in indices], dtype=np.int64)
    idxpath = OUT/'02_bootstrap_indices_seed20260924.npz'
    if idxpath.exists():
        with np.load(idxpath, allow_pickle=False) as stored:
            assert np.array_equal(stored['indices'], indices)
            assert np.array_equal(stored['multiplicities'], weights)
            assert np.array_equal(stored['sample_ids'], eq['sample_ids'])
    else:
        np.savez(idxpath, indices=indices, multiplicities=weights, labels=labels,
                 sample_ids=np.asarray(eq['sample_ids']), seed=SEED)
    protected = read(r1.OUT/'original_hashes.json')
    for path, digest in protected.items():
        assert sha(path)==digest, ('SOURCE_CHANGED',path)
    for path in [ROOT/'src/evaluation/weighted_bootstrap_exact.py', Path(__file__),
                 r1.OUT/'completion_verification.json', r1.OUT/'09_unit_tests.json', idxpath]:
        protected[str(path)] = sha(path)
    caches=[]
    for m, method in enumerate(METHODS):
        rows=inventory[method]['rows']
        assert [r['sample_id'] for r in rows]==eq['sample_ids']
        assert [r['gt_label'] for r in rows]==eq['labels']
        cache=read(r1.OUT/f'caches/m{m}/manifest.json')
        assert cache['sample_ids']==eq['sample_ids'] and cache['exact_aupro']
        assert sum(cache['region_counts'])==450
        caches.append(cache)
        for path in Path(cache['directory']).iterdir():
            if path.is_file(): protected[str(path)]=sha(path)
    for path in (r1.OUT/'gt_cache').iterdir():
        if path.is_file(): protected[str(path)]=sha(path)
    hashfile=OUT/'frozen_hashes.json'
    if hashfile.exists(): assert read(hashfile)==protected, 'RESUME_SOURCE_IDENTITY_CHANGED'
    else: save('frozen_hashes.json',protected)
    protocol=dict(phase='Phase21B-B',bootstrap_replicates=B,bootstrap_seed=SEED,
        bootstrap_unit='native_image',stratified=True,paired=True,normal_count=24,anomaly_count=90,
        methods=METHODS,baseline_count=5,ci_method='percentile',percentiles=[2.5,97.5],
        percentile_interpolation='linear',indices_sha256=sha(idxpath),
        predictions_read_only=True,bootstrap_duplicates_materialized=False,
        exact_pixel_auroc=True,exact_aupro005=True,exact_aupro030=True,
        gt_region_cache_created=True,total_gt_region_count=450,
        pixel_subsampling_used=False,threshold_approximation_used=False,gt_region_modified=False,
        image_score_scope='Method-level pipeline comparison; scoring grids/operators differ',
        fraction_delta_gt_zero_is_p_value=False,GPU_NOT_USED_FOR_BOOTSTRAP=True)
    if (OUT/'01_protocol_snapshot.json').exists(): assert read(OUT/'01_protocol_snapshot.json')==protocol
    else: save('01_protocol_snapshot.json',protocol)
    save('03_prediction_inventory.json',inventory)
    observed=read(r1.OLD/'04_observed_delta_inventory.json')['metrics']
    save('04_observed_metrics.json',observed)
    save('05_observed_deltas.json',{method:{k:100*(observed[METHODS[0]][k]-observed[method][k])
        for k in METRICS} for method in METHODS[1:]})
    session=dict(preparation_sec=time.perf_counter()-begin,cache_rebuilt=False,
                 timestamp=datetime.datetime.now().isoformat())
    history=read(OUT/'setup_sessions.json') if (OUT/'setup_sessions.json').exists() else []
    history.append(session);save('setup_sessions.json',history)
    print('PHASE21B_B_PREFLIGHT_PASS', session, flush=True)
    return protocol, inventory, labels, indices, weights, caches

def progress(jobs, active_extra=0):
    elapsed=sum(j['wall_sec'] for j in jobs)+active_extra
    count=len(jobs)
    value=dict(status='BOOTSTRAP_RUNNING',completed_replicates=count,total_replicates=B,
        elapsed_time_sec=elapsed,estimated_remaining_time_sec=(elapsed/count*(B-count) if count else None),
        current_peak_ram_gib=max([r1.old_driver.peak_ram()/1024]+[j['peak_ram_gib'] for j in jobs]),
        error_count=0,timestamp=datetime.datetime.now().isoformat())
    save('progress.json',value)
    return value

def finish(protocol, jobs):
    assert len(jobs)==B and [j['replicate'] for j in jobs]==list(range(B))
    values=np.asarray([[[result[k] for k in METRICS] for result in j['results']] for j in jobs])
    assert values.shape==(B,6,4) and np.isfinite(values).all()
    delta=values[:,0:1,:]-values[:,1:,:]
    np.savez(OUT/'06_bootstrap_raw_deltas.npz',delta_fraction=delta,delta_pp=delta*100,
             method_metrics_fraction=values,methods=np.asarray(METHODS),baselines=np.asarray(METHODS[1:]),
             metrics=np.asarray(METRICS),replicate_indices=np.arange(B))
    observed=read(OUT/'04_observed_metrics.json')
    table=[]
    for b, method in enumerate(METHODS[1:]):
        for k, metric in enumerate(METRICS):
            table.append(dict(baseline=method,metric=metric,
                **summarize(delta[:,b,k],observed[METHODS[0]][metric]-observed[method][metric])))
    save('07_bootstrap_summary.json',table)
    for metric,filename in zip(METRICS,['08_image_auroc_ci.json','09_pixel_auroc_ci.json','10_aupro005_ci.json','11_aupro030_ci.json']):
        save(filename,[r for r in table if r['metric']==metric])
    with (OUT/'13_full_ci_table.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    main=[]
    for method in METHODS[1:]:
        row={'comparison':'PB-ARPR vs '+method}
        for r in table:
            if r['baseline']==method and r['metric']!='image_auroc':
                for key in ['observed_delta_pp','ci_low_pp','ci_high_pp']:
                    row[r['metric']+'_'+key]=r[key]
        main.append(row)
    with (OUT/'12_main_localization_ci_table.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(main[0]));writer.writeheader();writer.writerows(main)
    for path,digest in read(OUT/'frozen_hashes.json').items():
        assert sha(path)==digest,('FROZEN_SOURCE_CHANGED',path)
    elapsed=sum(j['wall_sec'] for j in jobs)
    runtime=dict(cache_preparation_time_sec=sum(x['preparation_sec'] for x in read(OUT/'setup_sessions.json')),
        cache_preparation_scope='Existing read-only cache loading/hash verification; no rank cache rebuilding',
        bootstrap_wall_time_sec=elapsed,bootstrap_wall_time_hours=elapsed/3600,
        mean_replicate_time_sec=elapsed/B,peak_host_ram_gib=max(j['peak_ram_gib'] for j in jobs),
        GPU_NOT_USED_FOR_BOOTSTRAP=True,wall_time_scope='Sum of complete replicate wall intervals; resume downtime excluded')
    save('14_runtime_memory.json',runtime)
    integrity=dict(bootstrap_completed=B,failed_replicates=0,method_metric_count=24000,
        paired_delta_count=20000,all_finite=True,unique_replicates=B,original_hashes_unchanged=True)
    save('15_integrity_audit.json',integrity)
    audit=dict(protocol_violation_count=0,training_count=0,forward_count=0,
        original_predictions_unchanged=True,metric_implementation_unchanged=True,
        new_files=['scripts/run_phase21b_b.py','tests/test_phase21b_b.py',str(OUT)])
    save('16_protocol_audit.json',audit)
    completion=dict(**protocol,status='PASS',**runtime,**integrity,comparisons=table,protocol_violation_count=0)
    save('completion_verification.json',completion)
    lines=['PHASE21B-B FORMAL PAIRED STRATIFIED BOOTSTRAP','STATUS: PASS',
           f'Completed: {B}/{B}; seed={SEED}; 24 normal +90 anomaly; native image unit',
           f'Wall time: {elapsed:.6f} sec ({elapsed/3600:.6f} h)',
           f'Peak RAM: {runtime["peak_host_ram_gib"]:.6f} GiB',
           'All values below are percentage points; fraction(delta>0) is NOT a p-value.']
    for r in table:
        lines.append(f'{r["baseline"]} | {r["metric"]} | observed={r["observed_delta_pp"]:+.6f} | mean={r["bootstrap_mean_delta_pp"]:+.6f} | median={r["bootstrap_median_delta_pp"]:+.6f} | CI=[{r["ci_low_pp"]:+.6f}, {r["ci_high_pp"]:+.6f}] | fraction>0={r["fraction_delta_gt_zero"]:.6f}')
    lines+=['Image AUROC: method-level pipeline comparison; image scoring pipelines differ.',
            'Failed replicates:0; protocol violations:0; training:0; forward:0; no2000 run.',
            'New orchestration: scripts/run_phase21b_b.py; tests/test_phase21b_b.py']
    (OUT/'final_console_report.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save('progress.json',dict(status='PASS',completed_replicates=B,elapsed_time_sec=elapsed,
        estimated_remaining_time_sec=0,error_count=0,current_peak_ram_gib=runtime['peak_host_ram_gib']))
    print('\n'.join(lines),flush=True)

def run(precheck=False):
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'completion_verification.json').exists():
        assert read(OUT/'completion_verification.json')['status']!='PASS','Already completed; do not rerun'
    protocol,inventory,labels,indices,weights,caches=prepare()
    if precheck:return
    if (OUT/'failure.json').exists(): raise RuntimeError('Prior failure requires investigation; no automatic skipping/retry')
    scores=[np.array([r['image_score'] for r in inventory[m]['rows']]) for m in METHODS]
    jobs=[]
    for rep in range(B):
        path=OUT/f'replicates/{rep:04d}.json'
        if path.exists():
            job=read(path);validate_job(job,rep,protocol['indices_sha256']);jobs.append(job)
            continue
        begin=time.perf_counter();results=[]
        try:
            for m in range(6):
                start=time.perf_counter()
                image=float(roc_auc_score(labels[indices[rep]],scores[m][indices[rep]]))
                image_sec=time.perf_counter()-start
                result=exact.evaluate(caches[m],weights[rep])
                results.append(dict(method=METHODS[m],image_auroc=image,image_sec=image_sec,**result))
            job=dict(replicate=rep,indices_sha256=protocol['indices_sha256'],results=results,
                wall_sec=time.perf_counter()-begin,peak_ram_gib=r1.old_driver.peak_ram()/1024)
            save(path.relative_to(OUT),job);jobs.append(job)
            p=progress(jobs)
            if (rep+1)%50==0 or rep==0:print('PROGRESS',json.dumps(p),flush=True)
        except Exception:
            save('failure.json',dict(status='FAIL_BOOTSTRAP_REPLICATE',replicate=rep,
                traceback=traceback.format_exc(),failed_replicates=1))
            raise
    finish(protocol,jobs)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--precheck',action='store_true')
    run(parser.parse_args().precheck)
