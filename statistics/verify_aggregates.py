"""Read aggregate records only; no model/data access or training."""
from pathlib import Path
import csv,json,statistics,math
R=Path(__file__).resolve().parents[1]
def rows(p):return list(csv.DictReader((R/p).open(encoding='utf-8-sig')))
def read(p):return json.loads((R/p).read_text(encoding='utf-8-sig'))
u=rows('experiments_new/pbarpr_submission_audit_r1/image_auroc_unified.csv')
expected=read('experiments_new/pbarpr_submission_audit_r1/evaluation_summary.json')['summary']
series={g:{int(x['seed']):float(x['unified_image_auroc']) for x in u if x['geometry']==g} for g in ['square512','pb_arpr']}
out={}
for g,values in series.items():
 assert set(values)==set(range(42,47))
 a=list(values.values());m=statistics.mean(a);s=statistics.stdev(a)
 assert math.isclose(m,expected[g]['mean'],abs_tol=1e-12) and math.isclose(s,expected[g]['std'],abs_tol=1e-12)
 out[g]={'mean_percent':100*m,'sample_sd_percent':100*s}
d=[100*(series['pb_arpr'][s]-series['square512'][s]) for s in range(42,47)]
assert math.isclose(statistics.mean(d),expected['paired_delta_pp']['mean'],abs_tol=1e-12)
assert math.isclose(statistics.stdev(d),expected['paired_delta_pp']['std'],abs_tol=1e-12)
out['paired_delta_pp']={'mean':statistics.mean(d),'sample_sd':statistics.stdev(d),'positive_seeds':sum(v>0 for v in d)}
# KSDD2 per-seed aggregate metrics agree with the frozen official report.
k=rows('experiments/phase15e_ksdd2_blind_test/per_seed_metrics.csv')
ref=read('experiments/phase15e_ksdd2_blind_test/official_test_report.json')['metrics_by_seed']
for r in k:
 for metric in ['image_auroc','pixel_auroc','aupro005','aupro030']:
  assert math.isclose(float(r[metric]),ref[r['seed']][r['method']][metric],abs_tol=1e-12)
out['KSDD2_seed_records_verified']=len(k)
# Severstal: aggregate only; preserve negative image-AUROC difference.
s=read('experiments/phase17_severstal_final_blind/10_metric_comparison.json')
for metric,delta in s['delta_pp'].items():
 assert math.isclose(100*(s['metrics']['RECT'][metric]-s['metrics']['STD'][metric]),delta,abs_tol=1e-12)
assert s['delta_pp']['image_auroc']<0
out['Severstal_aggregate_deltas_verified']=True
out['status']='PASS_AGGREGATE_ARITHMETIC_ONLY'
print(json.dumps(out,indent=2))
