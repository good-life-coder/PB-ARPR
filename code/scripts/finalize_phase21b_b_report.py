"""Artifact-only verification/report recovery. No metric evaluation or forwards."""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_phase21b_b as formal

def main():
    out=formal.OUT
    read=formal.read
    protocol=read(out/'01_protocol_snapshot.json')
    runtime=read(out/'14_runtime_memory.json')
    integrity=read(out/'15_integrity_audit.json')
    table=read(out/'07_bootstrap_summary.json')
    observed=read(out/'04_observed_metrics.json')
    for path,digest in read(out/'frozen_hashes.json').items():
        assert formal.sha(path)==digest, ('FROZEN_HASH_CHANGED',path)
    jobs=[read(out/f'replicates/{r:04d}.json') for r in range(1000)]
    assert len(list((out/'replicates').glob('*.json')))==1000
    with np.load(out/'02_bootstrap_indices_seed20260924.npz') as idx:
        regenerated=formal.exact.bootstrap_indices(idx['labels'],1000,20260924)
        assert np.array_equal(regenerated,idx['indices'])
        assert np.all(np.sum(idx['labels'][regenerated]==0,axis=1)==24)
        assert np.all(np.sum(idx['labels'][regenerated]==1,axis=1)==90)
        assert np.array_equal(idx['multiplicities'],[np.bincount(i,minlength=114) for i in regenerated])
    for r,job in enumerate(jobs):
        formal.validate_job(job,r,protocol['indices_sha256'])
        assert [x['method'] for x in job['results']]==formal.METHODS
    values=np.asarray([[[x[k] for k in formal.METRICS] for x in j['results']] for j in jobs])
    assert values.shape==(1000,6,4) and np.isfinite(values).all()
    delta=values[:,0:1,:]-values[:,1:,:]
    with np.load(out/'06_bootstrap_raw_deltas.npz') as raw:
        assert np.array_equal(values,raw['method_metrics_fraction'])
        assert np.array_equal(delta,raw['delta_fraction'])
        assert np.array_equal(delta*100,raw['delta_pp'])
    for row in table:
        b=formal.METHODS.index(row['baseline'])-1
        k=formal.METRICS.index(row['metric'])
        d=delta[:,b,k]
        expected={'observed_delta_pp':100*(observed['PB-ARPR'][row['metric']]-observed[row['baseline']][row['metric']]),
                  'bootstrap_mean_delta_pp':100*float(d.mean()),
                  'bootstrap_median_delta_pp':100*float(np.median(d)),
                  'ci_low_pp':100*float(np.percentile(d,2.5,method='linear')),
                  'ci_high_pp':100*float(np.percentile(d,97.5,method='linear')),
                  'fraction_delta_gt_zero':float((d>0).mean())}
        for key,value in expected.items(): assert abs(row[key]-value)<1e-12
    assert abs(sum(j['wall_sec'] for j in jobs)-runtime['bootstrap_wall_time_sec'])<1e-9
    assert integrity['failed_replicates']==0 and len(table)==20
    recovery=dict(issue='Final report dict constructor received duplicate GPU_NOT_USED_FOR_BOOTSTRAP key',
        scope='Report assembly only; preserved original runner and all numerical artifacts',
        bootstrap_recomputed=False,metric_implementation_changed=False,
        new_file=str(Path(__file__).resolve()),independent_verification_pass=True)
    formal.save('report_recovery_audit.json',recovery)
    completion={**protocol,**runtime,**integrity,'status':'PASS','comparisons':table,
                'protocol_violation_count':0,'report_recovery':recovery}
    formal.save('completion_verification.json',completion)
    lines=['PHASE21B-B FORMAL PAIRED STRATIFIED BOOTSTRAP','STATUS: PASS',
           '1000/1000; failed replicates=0; seed=20260924; paired native-image unit;24 normal+90 anomaly',
           f'Bootstrap wall seconds: {runtime["bootstrap_wall_time_sec"]:.6f}',
           f'Peak RAM GiB: {runtime["peak_host_ram_gib"]:.6f}',
           'All deltas and CI bounds below in pp. Fraction(delta>0) is NOT a p-value.']
    for row in table:
        lines.append(f'{row["baseline"]} | {row["metric"]} | observed={row["observed_delta_pp"]:+.6f} | mean={row["bootstrap_mean_delta_pp"]:+.6f} | median={row["bootstrap_median_delta_pp"]:+.6f} | CI=[{row["ci_low_pp"]:+.6f},{row["ci_high_pp"]:+.6f}] | fraction>0={row["fraction_delta_gt_zero"]:.6f}')
    lines+=['Image AUROC is method-level pipeline comparison; scoring definitions differ.',
            'Protocol violations=0; no training; no forwards; no2000 run.',
            'Report-only duplicate-key failure recovered from saved results; no bootstrap rerun.',
            'New files: scripts/run_phase21b_b.py; tests/test_phase21b_b.py; scripts/finalize_phase21b_b_report.py']
    (out/'final_console_report.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    formal.save('progress.json',dict(status='PASS',completed_replicates=1000,total_replicates=1000,
        elapsed_time_sec=runtime['bootstrap_wall_time_sec'],estimated_remaining_time_sec=0,
        current_peak_ram_gib=runtime['peak_host_ram_gib'],error_count=0,report_only_error_recovered=True))
    print('INDEPENDENT_VERIFICATION_PASS;1000 unique replicates;20 summaries; all frozen hashes unchanged')

if __name__=='__main__':main()
