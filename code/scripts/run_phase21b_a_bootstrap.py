"""Read-only frozen-prediction audit and exactly 20 paired image bootstrap jobs.

No model code is imported. Exact AUPRO is imported unchanged; its score loader
must never resize because every input is already a verified native map.
"""
import argparse
import csv
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'experiments/phase21b_a_bootstrap_preflight'
EXP = ROOT / 'experiments'
DATA = ROOT / 'datasets/MVTec_AD_2/sheet_metal'
SHAPE = (1056, 4224)
METHODS = ['PB-ARPR', 'EfficientAD-S', 'Square512', 'Tile4', 'PatchCore', 'FastFlow']
SEED = 20260924
EXACT_SHA = '3e6ebc80d400419cf3aaf00950e954ea165df2419c6ea895f4296013c67e11a1'
METRIC = ROOT / 'src/evaluation/exact_aupro.py'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def peak_ram():
    class PMC(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)] + [
            (n, ctypes.c_size_t) for n in ['PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
            'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage',
            'PagefileUsage', 'PeakPagefileUsage']]
    x = PMC()
    x.cb = ctypes.sizeof(x)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(x), x.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return x.PeakWorkingSetSize / 1024**2


def csvrows(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def canon(path):
    return Path(path).resolve().relative_to(DATA.resolve()).as_posix()


def indices(labels):
    import numpy as np
    rng = np.random.default_rng(SEED)
    y = np.asarray(labels)
    normal = np.flatnonzero(y == 0)
    anomaly = np.flatnonzero(y == 1)
    assert len(normal) == 24 and len(anomaly) == 90
    return [np.r_[rng.choice(normal, 24, replace=True), rng.choice(anomaly, 90, replace=True)].tolist()
            for _ in range(20)]


def audit():
    import numpy as np
    from PIL import Image
    assert sha(METRIC) == EXACT_SHA, 'Frozen exact metric hash mismatch'
    assert not (OUT / 'audit_complete.json').exists(), 'Audit already frozen; use run mode'
    source_hashes = {}

    def tracked(p):
        p = Path(p)
        source_hashes[str(p)] = sha(p)
        return p

    std = EXP / 'phase07c_baseline256_public'
    sq = EXP / 'phase18b_sheet_metal_square512_70k_formal'
    pb = EXP / 'phase13c_rectangular_256x1024_public'
    tile = EXP / 'phase19b_sheet_metal_tile4_70k_formal'
    pc = EXP / 'phase20b_p2_patchcore_formal'
    ff = EXP / 'phase20c_p2_fastflow_formal'
    allrows = {
        'PB-ARPR': csvrows(tracked(pb / 'predictions.csv')),
        'EfficientAD-S': csvrows(tracked(std / 'predictions.csv')),
        'Square512': csvrows(tracked(sq / 'evaluation/predictions.csv')),
        'Tile4': read(tracked(tile / 'native_predictions.json')),
        'PatchCore': read(tracked(pc / 'prediction_manifest.json')),
        'FastFlow': read(tracked(ff / 'prediction_manifest.json')),
    }
    square_manifest = {r['sample_id']: r for r in read(tracked(sq / 'native_manifest.json'))}
    stack = np.load(tracked(std / 'metrics/native_scores.npy'), mmap_mode='r')
    targets = np.load(tracked(std / 'metrics/native_targets.npy'), mmap_mode='r')
    assert stack.shape == targets.shape == (114, *SHAPE)
    assert stack.dtype == np.float32 and targets.dtype == np.uint8
    tracked(ROOT / 'scripts/evaluate_baseline256_public.py')
    tracked(METRIC)
    cache = OUT / 'readonly_native_stack_cache'
    cache.mkdir(exist_ok=True)
    normalized = {}
    inventories = {}
    mask_hashes = {}
    for method in METHODS:
        result = []
        rows = allrows[method]
        assert len(rows) == 114, (method, 'row count')
        for i, r in enumerate(rows):
            sid = canon(r['image_path'])
            assert sid.startswith(('test_public/good/', 'test_public/bad/'))
            label = int(r['gt_label'])
            assert label == int('/bad/' in sid)
            assert Path(r['image_path']).is_file()
            maskpath = str(Path(r['gt_mask_path']).resolve()) if label else ''
            if label:
                expected = DATA / 'test_public/ground_truth/bad' / (Path(r['image_path']).stem + '_mask.png')
                assert Path(maskpath) == expected.resolve(), (method, sid, 'GT path')
                if maskpath not in mask_hashes:
                    mask_hashes[maskpath] = sha(maskpath)
                with Image.open(maskpath) as im:
                    mask = (np.asarray(im.convert('L')) > 0).astype(np.uint8)
            else:
                mask = np.zeros(SHAPE, np.uint8)
            assert mask.shape == SHAPE
            if method == 'EfficientAD-S':
                assert np.array_equal(targets[i], mask), ('STD native stack order/GT mismatch', sid)
                arr = stack[i]
                path = cache / (r['sample_id'] + '.npy')
                if path.exists():
                    assert np.array_equal(np.load(path, mmap_mode='r'), arr)
                else:
                    np.save(path, arr, allow_pickle=False)
                assert np.array_equal(np.load(path, mmap_mode='r'), arr)
                expected_hash = None
                score = float(r['official_image_score'])
            elif method == 'Square512':
                entry = square_manifest[r['sample_id']]
                path = Path(entry['path'])
                expected_hash = entry['sha256']
                score = float(r['official_image_score'])
            elif method == 'PB-ARPR':
                path = Path(r['combined_path'])
                expected_hash = None
                score = float(r['combined_image_score'])
            elif method == 'Tile4':
                path = Path(r['raw_map_path'])
                expected_hash = r['map_sha256']
                score = float(r['image_score'])
            else:
                path = Path(r['raw_map_path'])
                expected_hash = r['native_sha256']
                score = float(r['official_image_score'])
            arr = np.load(path, mmap_mode='r', allow_pickle=False)
            assert arr.shape == SHAPE and arr.dtype == np.float32, (method, sid, arr.shape, arr.dtype)
            assert np.isfinite(arr).all() and np.isfinite(score), (method, sid, 'nonfinite')
            digest = sha(path)
            if expected_hash:
                assert digest == expected_hash, (method, sid, 'map hash mismatch')
            source_hashes[str(path)] = digest
            result.append(dict(sample_id=sid, image_path=str(Path(r['image_path']).resolve()),
                gt_label=label, gt_mask_path=maskpath, native_shape=list(SHAPE),
                raw_map_path=str(path.resolve()), image_score=score, map_sha256=digest,
                mask_sha256=mask_hashes.get(maskpath), source_row_index=i, source_sample_id=r['sample_id']))
        result.sort(key=lambda r: r['sample_id'])
        assert len({r['sample_id'] for r in result}) == 114
        normalized[method] = result
        inventories[method] = dict(image_score_count=114, native_map_count=114, shape=list(SHAPE),
            nan=0, inf=0, missing=0, duplicate=0, rows=result)
        print('AUDIT', method, '114 maps PASS', flush=True)
    base = normalized['PB-ARPR']
    for method, rows in normalized.items():
        for a, b in zip(base, rows):
            for field in ['sample_id', 'image_path', 'gt_label', 'gt_mask_path', 'mask_sha256']:
                assert a[field] == b[field], (method, field, a['sample_id'])
    assert len(mask_hashes) == 90
    source_hashes.update(mask_hashes)
    labels = [r['gt_label'] for r in base]
    draws = indices(labels)
    save('01_prediction_inventory.json', inventories)
    save('02_paired_alignment.json', dict(paired_sample_alignment_pass=True, native_sample_count=114,
        normal_count=24, anomaly_count=90, mask_count=90, missing=0, duplicate=0,
        path_mismatch=0, label_mismatch=0, gt_mismatch=0,
        std_stack_order_evidence='evaluate_baseline256_public.py:387-395; every stored target checked against CSV mask',
        std_cache='Bit-identical existing native stack slices; no prediction or resize; source stack unchanged'))
    protocol = dict(bootstrap_unit='native_image', stratified=True, paired=True, normal_draw_count=24,
        anomaly_draw_count=90, replacement=True, multiplicity='Retain every draw, virtual unique instance ID',
        calibration_replicates=20, formal_replicates=1000, bootstrap_seed=SEED, rng='numpy.default_rng/PCG64',
        ci_method='percentile', ci_lower_percentile=2.5, ci_upper_percentile=97.5,
        percentile_method='linear', delta='PB-ARPR minus baseline', fraction_positive_is_p_value=False,
        baselines=METHODS[1:], metric_implementation_unchanged=True, exact_metric_sha256=EXACT_SHA,
        pixel_auroc='sklearn.metrics.roc_auc_score on all pixels from every selected native image occurrence',
        exact_aupro='src.evaluation.exact_aupro.native; both caps jointly calculated',
        image_score_scope='Method-level comparison: model-grid max (STD/Square/FastFlow); native max (PB/Tile); official weighted NN (PatchCore)',
        gpu_used=False, training=False, model_forward_count=0)
    save('03_bootstrap_protocol.json', protocol)
    frozen = {}
    m = read(tracked(pb / 'combined_metrics.json'))
    frozen['PB-ARPR'] = {k: m[k] for k in ['image_auroc', 'pixel_auroc', 'aupro005', 'aupro030']}
    m = read(tracked(std / 'evaluation_report.json'))['metrics']
    a = read(tracked(EXP / 'phase07c_r1_aupro_exact/aupro_report.json'))
    frozen['EfficientAD-S'] = dict(image_auroc=m['image_auroc'], pixel_auroc=m['pixel_auroc'],
        aupro005=a['aupro_0.05'], aupro030=a['aupro_0.30'])
    for method, folder in [('Square512', sq), ('Tile4', tile), ('PatchCore', pc), ('FastFlow', ff)]:
        m = read(tracked(folder / 'completion_verification.json'))
        assert m['status'] == 'PASS'
        frozen[method] = dict(image_auroc=m['image_auroc'], pixel_auroc=m['pixel_auroc'],
            aupro005=m['exact_aupro005'], aupro030=m['exact_aupro030'])
    save('04_observed_delta_inventory.json', dict(historical_metrics_read_only=True, metrics=frozen,
        delta_pp={m: {k: 100*(frozen['PB-ARPR'][k]-frozen[m][k]) for k in frozen[m]} for m in METHODS[1:]}))
    save('05_calibration_indices.json', dict(seed=SEED, sample_ids=[r['sample_id'] for r in base], labels=labels,
        indices=draws, multiplicities=[np.bincount(d, minlength=114).tolist() for d in draws]))
    save('source_hashes.json', source_hashes)
    save('audit_complete.json', dict(status='PREFLIGHT_PASS', peak_host_ram_mib=peak_ram(),
        inventory_sha256=sha(OUT/'01_prediction_inventory.json'), indices_sha256=sha(OUT/'05_calibration_indices.json'),
        source_hashes_sha256=sha(OUT/'source_hashes.json')))


class NoResize:
    """Fail closed if the frozen score loader ever tries a new resize."""
    def __getattr__(self, name):
        raise RuntimeError('Native map required; no torch operations permitted: '+name)


def worker(rep, method_index):
    import numpy as np
    import cv2
    from PIL import Image
    from sklearn.metrics import roc_auc_score
    from src.evaluation import exact_aupro
    assert sha(METRIC) == EXACT_SHA
    method = METHODS[method_index]
    done = OUT / f'jobs/r{rep:02d}_m{method_index}.json'
    assert not done.exists(), 'Completed job must not be overwritten'
    audit_info = read(OUT / 'audit_complete.json')
    assert sha(OUT/'01_prediction_inventory.json') == audit_info['inventory_sha256']
    assert sha(OUT/'05_calibration_indices.json') == audit_info['indices_sha256']
    inventory = read(OUT / '01_prediction_inventory.json')[method]['rows']
    selected = read(OUT / '05_calibration_indices.json')['indices'][rep]
    rows = [dict(inventory[idx], native_shape=SHAPE, sample_id=f'draw{j:03d}::{inventory[idx]["sample_id"]}')
            for j, idx in enumerate(selected)]
    assert len(rows) == 114 and sum(r['gt_label'] for r in rows) == 90
    begin = time.perf_counter()
    t = time.perf_counter()
    image_auc = float(roc_auc_score([r['gt_label'] for r in rows], [r['image_score'] for r in rows]))
    image_sec = time.perf_counter()-t
    tmp = OUT / 'work'
    tmp.mkdir(exist_ok=True)
    t = time.perf_counter()
    scores = np.lib.format.open_memmap(tmp/'scores.npy', mode='w+', dtype=np.float32, shape=(114,*SHAPE))
    targets = np.lib.format.open_memmap(tmp/'targets.npy', mode='w+', dtype=np.uint8, shape=(114,*SHAPE))
    for i,r in enumerate(rows):
        scores[i] = np.load(r['raw_map_path'], mmap_mode='r', allow_pickle=False)
        targets[i] = exact_aupro.get_mask(r, Image, np)
    scores.flush(); targets.flush()
    pixel_auc = float(roc_auc_score(targets.reshape(-1), scores.reshape(-1)))
    pixel_sec = time.perf_counter()-t
    del scores, targets
    gc.collect()
    t = time.perf_counter()
    report = {}
    exact_aupro.native(rows, report, NoResize(), cv2, np, Image, temp_dir=tmp/'exact')
    exact_sec = time.perf_counter()-t
    result = dict(replicate=rep, method=method, indices_sha256=audit_info['indices_sha256'],
        image_auroc=image_auc, pixel_auroc=pixel_auc, aupro005=report['aupro_0.05'], aupro030=report['aupro_0.30'],
        image_sec=image_sec, pixel_sec=pixel_sec, exact_joint_sec=exact_sec,
        wall_sec=time.perf_counter()-begin, peak_host_ram_mib=peak_ram(),
        connected_regions=report['connected_regions'], background_pixels=report['background_pixels'],
        positive_pixels=report['positive_pixels'], weight_sum=report['weight_sum'], unique_images=len(set(selected)),
        native_image_draws=114, success=True)
    save(done.relative_to(OUT), result)
    print('JOB_COMPLETE', rep, method, result['wall_sec'], flush=True)


def finish(wall):
    jobs = [read(OUT/f'jobs/r{r:02d}_m{m}.json') for r in range(20) for m in range(6)]
    assert len(jobs) == 120 and all(j['success'] for j in jobs)
    for j in jobs:
        assert j['indices_sha256'] == read(OUT/'audit_complete.json')['indices_sha256']
        assert j['native_image_draws'] == 114
    paired_results = []
    for r in range(20):
        by_method = {j['method']:j for j in jobs if j['replicate']==r}
        for baseline in METHODS[1:]:
            paired_results.append(dict(replicate=r, baseline=baseline,
                pb_arpr={k:by_method['PB-ARPR'][k] for k in ['image_auroc','pixel_auroc','aupro005','aupro030']},
                baseline_metrics={k:by_method[baseline][k] for k in ['image_auroc','pixel_auroc','aupro005','aupro030']},
                delta_pp={k:100*(by_method['PB-ARPR'][k]-by_method[baseline][k])
                          for k in ['image_auroc','pixel_auroc','aupro005','aupro030']}))
    save('calibration_paired_results.json',paired_results)
    assert sha(METRIC) == EXACT_SHA
    for p,h in read(OUT/'source_hashes.json').items():
        assert sha(p) == h, ('Source changed', p)
    joint = sum(j['exact_joint_sec'] for j in jobs)
    runtime = dict(calibration_success_count=20, method_metric_jobs=120,
        image_auroc_calibration_time_sec=sum(j['image_sec'] for j in jobs),
        pixel_auroc_calibration_time_sec=sum(j['pixel_sec'] for j in jobs),
        aupro005_calibration_time_sec=joint, aupro030_calibration_time_sec=joint,
        aupro_joint_time_sec=joint, aupro_timing_shared=True,
        timing_note='Both exact AUPRO caps are returned by one unchanged call; shared times must not be added twice.',
        total_20_wall_time_sec=wall, jobs=jobs)
    save('06_runtime_calibration.json', runtime)
    peak = max([read(OUT/'audit_complete.json')['peak_host_ram_mib']]+[j['peak_host_ram_mib'] for j in jobs])
    save('07_memory_audit.json', dict(peak_host_ram_mib=peak, measurement='Windows PeakWorkingSetSize per serial subprocess',
        strategy='One method/replicate at a time; read-only original maps; disk-backed score/target/exact arrays',
        simultaneous_methods=1, gpu_used=False))
    estimates = {f'estimated_{n}_time': wall*n/20 for n in [500,1000,2000]}
    save('08_formal_runtime_estimate.json', dict(**estimates, unit='seconds', estimated_not_measured=True,
        method='Linear extrapolation of 20 complete six-method replicates, including worker startup',
        caveat='Exact sort runtime depends on replicate score/region distribution and OS file cache; extrapolation is not guaranteed.'))
    authorized = estimates['estimated_1000_time'] <= 4*3600
    status = 'PASS_FOR_BOOTSTRAP_1000' if authorized else 'HOLD_BOOTSTRAP_RUNTIME'
    protocol = read(OUT/'03_bootstrap_protocol.json')
    save('09_phase21b_b_preregistration.json', dict(**protocol, status=status, phase21b_b_authorized=authorized,
        formal_run_started=False, observed_delta_source='04_observed_delta_inventory.json',
        required_outputs=['observed_delta','bootstrap_mean_delta','bootstrap_median_delta','ci_low','ci_high','fraction_delta_positive']))
    completion = dict(phase='Phase21B-A', status=status, native_sample_count=114, normal_count=24, anomaly_count=90,
        methods=METHODS, paired_sample_alignment_pass=True, **protocol,
        calibration_success_count=20, **{k:v for k,v in runtime.items() if k.endswith('_sec')}, **estimates,
        peak_host_ram_mib=peak, phase21b_b_authorized=authorized, protocol_violation_count=0,
        historical_sources_hash_unchanged=True, prediction_regeneration_count=0)
    save('completion_verification.json', completion)
    lines = ['PHASE21B-A PAIRED BOOTSTRAP PREFLIGHT', f'STATUS: {status}',
        'Alignment: 6 methods x 114 PASS; 24 normal / 90 anomaly; all native maps 1056x4224 finite',
        'Native-image stratified paired bootstrap; replacement; normal24/anomaly90; seed20260924',
        'Calibration: 20/20 replicates, 120/120 method metric jobs',
        f'Total 20 wall time: {wall:.6f} sec',
        f'Image AUROC: {runtime["image_auroc_calibration_time_sec"]:.6f} sec',
        f'Pixel AUROC: {runtime["pixel_auroc_calibration_time_sec"]:.6f} sec',
        f'Exact AUPRO005/030 shared: {joint:.6f} sec (joint, not additive)',
        *[f'{k}: {v/3600:.6f} h ESTIMATED_NOT_MEASURED' for k,v in estimates.items()],
        f'Peak host RAM: {peak:.3f} MiB', f'PHASE21B_B_AUTHORIZED: {authorized}',
        'Metric implementation unchanged; historical source hashes unchanged; no forward/training; no formal1000 run',
        'Image AUROC is a method-level pipeline comparison; retain all original score definitions.',
        'Protocol violations: 0']
    (OUT/'final_console_report.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines), flush=True)


def run():
    assert (OUT/'audit_complete.json').exists()
    assert not (OUT/'completion_verification.json').exists(), 'Final report exists'
    elapsed = read(OUT/'progress.json').get('accumulated_wall_sec',0) if (OUT/'progress.json').exists() else 0
    start = time.perf_counter()
    for rep in range(20):
        for m, method in enumerate(METHODS):
            if (OUT/f'jobs/r{rep:02d}_m{m}.json').exists():
                continue
            print(f'START replicate {rep+1}/20 method {method}', flush=True)
            logfile = OUT/f'logs/r{rep:02d}_m{m}.txt'
            logfile.parent.mkdir(exist_ok=True)
            with logfile.open('a',encoding='utf-8') as log:
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'worker','--rep',str(rep),'--method',str(m)],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            save('progress.json',dict(completed_jobs=len(list((OUT/'jobs').glob('*.json'))),
                accumulated_wall_sec=elapsed+time.perf_counter()-start, last_replicate=rep+1,last_method=method))
    finish(elapsed+time.perf_counter()-start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=['audit','run','worker'])
    parser.add_argument('--rep',type=int)
    parser.add_argument('--method',type=int)
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    try:
        if args.mode == 'audit': audit()
        elif args.mode == 'run': run()
        else: worker(args.rep,args.method)
    except Exception:
        traceback.print_exc()
        if args.mode == 'audit':
            save('completion_verification.json',dict(phase='Phase21B-A',status='FAIL_PAIRED_ALIGNMENT',
                error=traceback.format_exc(),phase21b_b_authorized=False,calibration_success_count=0,
                model_forward_count=0,training=False))
            (OUT/'final_console_report.txt').write_text('STATUS: FAIL_PAIRED_ALIGNMENT\n'+traceback.format_exc(),encoding='utf-8')
        raise
