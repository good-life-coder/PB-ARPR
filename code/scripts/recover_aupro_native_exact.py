"""Exact CPU AU-PRO recovery from immutable saved predictions; no model imports.

Default: synthetic -> low-resolution control -> native calculation.
--synthetic-only: small mathematical/API checks only.
--validate-only: synthetic and low-resolution control, no native calculation.
"""

import argparse
import csv
import gc
import hashlib
import inspect
import json
import math
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.evaluation.exact_aupro import integrate, components, get_mask, get_score, native
SOURCE = ROOT / "experiments/phase07c_baseline256_public"
OUT = ROOT / "experiments/phase07c_r1_aupro_exact"
TEMP = OUT / "temp"
ALPHAS = (0.05, 0.30)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 ** 2), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def initial_report():
    return dict(dataset="MVTec AD 2", category="sheet_metal", split="test_public", status="NOT_RUN",
                algorithm="exact rank-based native-resolution PRO integration", approximation=False,
                test_count=None, normal_count=None, anomaly_count=None, raw_map_shape=None,
                native_resolution=None, shape_order="height,width", interpolation_method="bilinear; align_corners=False; only if needed",
                background_pixels=None, positive_pixels=None, connected_regions=None, weight_sum=None,
                **{"aupro_0.05": None, "aupro_0.30": None, "aupro_0.05_percent": None, "aupro_0.30_percent": None},
                image_auroc_percent=61.57, pixel_auroc_percent=79.54,
                existing_metrics_source="phase07c_baseline256_public", existing_metrics_recomputed=False,
                background_dtype="float32", sorting_method="numpy.memmap.sort(kind='quicksort'), ascending in-place",
                error_message=None, warnings=[])


def summary(report, validation):
    lines = ["=" * 40, "PHASE 07C-R1 - EXACT AU-PRO RECOVERY", "=" * 40,
             "Dataset: MVTec AD 2 / sheet_metal", "Split: test_public",
             f"Native resolution (H,W): {report['native_resolution']}",
             f"Test images: {report['test_count']}", f"Connected GT regions: {report['connected_regions']}",
             f"Background pixels: {report['background_pixels']}", f"Positive pixels: {report['positive_pixels']}",
             "\nLOW-RES VALIDATION"]
    for item in validation.get("lowres", []):
        lines.extend([f"Anomalib AU-PRO@{item['alpha']}: {100 * item['anomalib']:.4f} %",
                      f"Custom AU-PRO@{item['alpha']}: {100 * item['custom']:.4f} %",
                      f"Difference: {item['difference']:.8f}"])
    lines.extend([f"Validation: {validation.get('status', 'NOT_RUN')}", "\nNATIVE EXACT RESULTS",
                  "Image AUROC: 61.57 % (Phase 07C)", "Pixel AUROC: 79.54 % (Phase 07C)"])
    for alpha in ALPHAS:
        value = report[f"aupro_{alpha:.2f}"]
        lines.append(f"AU-PRO@{alpha:.2f}: " + ("N/A" if value is None else f"{100*value:.2f} %"))
    lines.extend(["Approximation: False", "STATUS:", report["status"]])
    if report["error_message"]:
        lines.append(report["error_message"])
    return "\n".join(lines + ["=" * 40])


def save_reports(report, validation):
    errors = []
    for name, value in (("validation_report.json", validation), ("aupro_report.json", report),
                        ("aupro_report.txt", summary(report, validation))):
        try:
            if name.endswith("json"):
                write_json(OUT / name, value)
            else:
                (OUT / name).write_text(value + "\n\n" + json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            traceback.print_exc()
            errors.append(str(exc))
    if errors:
        raise OSError("; ".join(errors))






def custom_control(predictions, masks, cv2, np):
    backgrounds, positives, weights = [], [], []
    k = 0
    for score, mask in zip(predictions, masks):
        labels, sizes = components(mask, cv2, np)
        fg = labels > 0
        backgrounds.append(score[~fg])
        positives.append(score[fg])
        weights.append(1.0 / sizes[labels[fg]])
        k += len(sizes) - 1
    bg = np.concatenate(backgrounds)
    bg.sort(kind="quicksort")
    values, total = integrate(bg, np.concatenate(positives), np.concatenate(weights) / k, np)
    if abs(total - 1) >= 1e-6:
        raise ValueError(f"Control region weights sum to {total}")
    return values


def compare(predictions, masks, AUPRO, torch, cv2, np):
    custom = custom_control(predictions, masks, cv2, np)
    results = []
    for alpha in ALPHAS:
        metric = AUPRO(fields=["anomaly_map", "gt_mask"], fpr_limit=alpha).cpu()
        for score, mask in zip(predictions, masks):
            metric.update(SimpleNamespace(anomaly_map=torch.from_numpy(score[None]),
                                          gt_mask=torch.from_numpy(mask[None])))
        actual = float(metric.compute().item())
        results.append(dict(alpha=alpha, anomalib=actual, custom=custom[alpha],
                            difference=abs(actual - custom[alpha])))
        metric.reset()
        del metric
        gc.collect()
    return results


def synthetic(AUPRO, torch, cv2, np):
    rng = np.random.default_rng(42)
    cases = []
    one = np.zeros((2, 12, 12), dtype=np.uint8)
    one[0, 2:5, 3:6] = 1
    many = one.copy()
    many[0, 8, 8] = 1
    many[1, 2:7, 2:7] = 1
    many[1, 9, 9] = many[1, 10, 10] = 1  # diagonal 8-connectivity
    for name, mask, score in (
        ("single_component", one, rng.random(one.shape).astype(np.float32)),
        ("unequal_multiple_regions_with_ties", many, (rng.integers(0, 5, many.shape) / 4).astype(np.float32)),
        ("all_scores_equal", many, np.full(many.shape, 0.5, dtype=np.float32)),
        ("perfect_ranking", many, many.astype(np.float32)),
    ):
        cases.append(dict(name=name, results=compare(score, mask, AUPRO, torch, cv2, np)))
    return cases


def acceptable(results):
    return all(math.isfinite(item["anomalib"]) and math.isfinite(item["custom"])
               and item["difference"] <= 1e-4 for item in results)


def read_inputs(np, Image, report):
    with (SOURCE / "predictions.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    raw_files = list((SOURCE / "raw_anomaly_maps").glob("*.npy"))
    if len(rows) != 114 or len(raw_files) != 114:
        raise ValueError("Expected exactly 114 CSV rows and raw maps")
    if [int(row["gt_label"]) for row in rows].count(0) != 24 or [int(row["gt_label"]) for row in rows].count(1) != 90:
        raise ValueError("Expected 24 normal and 90 anomalous images")
    seen, images, shapes, native_shapes = set(), set(), set(), set()
    manifest = []
    for row in rows:
        path = Path(row["raw_map_path"]).resolve()
        if path.parent != (SOURCE / "raw_anomaly_maps").resolve() or path in seen:
            raise ValueError("Raw map must be unique and belong to Phase 07C")
        seen.add(path)
        image_path = Path(row["image_path"]).resolve()
        public_root = (ROOT / "datasets/MVTec_AD_2/sheet_metal/test_public").resolve()
        if not image_path.is_relative_to(public_root) or image_path in images:
            raise ValueError("Images must be unique official public samples")
        images.add(image_path)
        with Image.open(image_path) as image:
            width, height = image.size
        if int(row["gt_label"]):
            mask_path = Path(row["gt_mask_path"]).resolve()
            if not mask_path.is_relative_to(public_root / "ground_truth"):
                raise ValueError("GT mask must be official public ground truth")
            with Image.open(mask_path) as mask:
                if mask.size != (width, height) or not np.asarray(mask).any():
                    raise ValueError(f"Invalid native mask: {mask_path}")
        raw = np.load(path, mmap_mode="r", allow_pickle=False)
        if raw.dtype != np.float32 or raw.ndim != 2 or not np.isfinite(raw).all():
            raise ValueError(f"Raw map must be finite float32 2D: {path}")
        if raw.shape not in ((256, 256), (height, width)):
            raise ValueError(f"Unsupported raw map shape: {raw.shape}")
        row["native_shape"] = (height, width)
        shapes.add(tuple(raw.shape))
        native_shapes.add((height, width))
        manifest.append(dict(path=str(path), sha256=digest(path)))
    if seen != {path.resolve() for path in raw_files}:
        raise ValueError("Raw map directory does not match CSV exactly")
    report.update(test_count=114, normal_count=24, anomaly_count=90,
                  raw_map_shape=sorted(shapes), native_resolution=sorted(native_shapes),
                  input_csv_sha256=digest(SOURCE / "predictions.csv"), input_raw_map_hashes=manifest)
    return rows








def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--synthetic-only", action="store_true")
    group.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    report = initial_report()
    validation = dict(status="NOT_RUN", tolerance=1e-4, preferred_tolerance=1e-5)
    started = time.perf_counter()
    report["timestamp"] = datetime.now().astimezone().isoformat()
    stage = "FAIL_OTHER"
    code = 1
    try:
        os.chdir(ROOT)
        TEMP.mkdir(parents=True, exist_ok=True)
        import numpy as np
        import torch
        import cv2
        import anomalib
        from PIL import Image
        from anomalib.metrics import AUPRO
        from anomalib.metrics.pro import connected_components_cpu

        actual_version = getattr(anomalib, "__version__", version("anomalib"))
        report["anomalib_version"] = actual_version
        if actual_version != "2.6.1":
            raise RuntimeError(f"Expected anomalib 2.6.1, found {actual_version}")
        implementation = next(cls for cls in AUPRO.__mro__ if cls.__name__ == "_AUPRO")
        helper_source = inspect.getsource(connected_components_cpu)
        validation.update(anomalib_version=actual_version, aupro_api=str(inspect.signature(AUPRO)),
                          aupro_implementation_api=str(inspect.signature(implementation)),
                          anomalib_aupro_source=inspect.getsource(implementation),
                          connected_component_source=helper_source,
                          connected_component_method="OpenCV cv2.connectedComponents", connectivity=8,
                          opencv_api_doc=cv2.connectedComponents.__doc__)
        if "cv2.connectedComponents(mask)" not in helper_source:
            raise RuntimeError("Installed connected-component helper changed; inspect connectivity before proceeding")
        diagonal = np.eye(2, dtype=np.uint8)
        assert components(diagonal, cv2, np)[1].size == 2
        official_diagonal = connected_components_cpu(torch.from_numpy(diagonal).float()[None, None])
        if int(official_diagonal.max()) != 1:
            raise RuntimeError("Official helper does not join diagonal pixels; connectivity requires review")
        validation["diagonal_connectivity_check"] = "PASS: official helper and custom implementation both join diagonal pixels"
        validation["synthetic"] = synthetic(AUPRO, torch, cv2, np)
        if not all(acceptable(case["results"]) for case in validation["synthetic"]):
            report["status"] = validation["status"] = "CHECK_REQUIRED"
            return 2
        validation["synthetic_status"] = "PASS"
        if args.synthetic_only:
            report["status"] = "NOT_RUN"
            validation["status"] = "SYNTHETIC_PASS_LOWRES_NOT_RUN"
            return 0
        stage = "FAIL_INPUT"
        rows = read_inputs(np, Image, report)
        original = json.loads((SOURCE / "metrics/public_metrics.json").read_text(encoding="utf-8"))
        report["existing_metrics_reference"] = {key: original.get(key) for key in ("image_auroc", "pixel_auroc")}
        stage = "CHECK_REQUIRED"
        scores, masks = [], []
        for row in rows:
            score = np.load(row["raw_map_path"], allow_pickle=False)
            if score.shape != (256, 256):
                score = torch.nn.functional.interpolate(torch.from_numpy(score)[None, None], size=(256, 256),
                                                       mode="bilinear", align_corners=False)[0, 0].numpy()
            mask = torch.from_numpy(get_mask(row, Image, np)).float()[None, None]
            mask = torch.nn.functional.interpolate(mask, size=(256, 256), mode="nearest")[0, 0].numpy().astype(np.uint8)
            scores.append(score)
            masks.append(mask)
        validation["lowres_shape"] = [256, 256]
        validation["lowres_gt_resize"] = "PyTorch nearest; diagnostic only, never used for native metric"
        validation["lowres"] = compare(scores, masks, AUPRO, torch, cv2, np)
        for item in validation["lowres"]:
            suffix = "005" if item["alpha"] == 0.05 else "030"
            report[f"lowres_control_anomalib_{suffix}"] = item["anomalib"]
            report[f"lowres_control_custom_{suffix}"] = item["custom"]
            report[f"lowres_difference_{suffix}"] = item["difference"]
        if not acceptable(validation["lowres"]):
            report["status"] = validation["status"] = "CHECK_REQUIRED"
            return 2
        validation["status"] = "PASS"
        validation["preferred_tolerance_met"] = all(item["difference"] <= 1e-5 for item in validation["lowres"])
        del scores, masks
        gc.collect()
        save_reports(report, validation)
        if args.validate_only:
            report["status"] = "VALIDATION_PASS_NATIVE_NOT_RUN"
            return 0
        stage = "FAIL_NATIVE"
        native(rows, report, torch, cv2, np, Image, temp_dir=TEMP)
        stage = "FAIL_INPUT"
        if digest(SOURCE / "predictions.csv") != report["input_csv_sha256"]:
            raise ValueError("Input CSV changed during recovery")
        for item in report["input_raw_map_hashes"]:
            if digest(Path(item["path"])) != item["sha256"]:
                raise ValueError(f"Raw map changed: {item['path']}")
        report["status"] = "PASS"
        # Retain memmaps by default for audit; no automatic deletion is necessary.
        report["temporary_files_retained"] = True
        code = 0
    except (Exception, KeyboardInterrupt) as exc:
        if isinstance(exc, ArithmeticError):
            stage = "FAIL_REGION_WEIGHT"
        report.update(status=stage, error_message=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        try:
            OUT.mkdir(parents=True, exist_ok=True)
            save_reports(report, validation)
        except Exception as exc:
            traceback.print_exc()
            report.update(status="FAIL_OTHER", error_message=f"Report writing failed: {exc}")
            code = 1
        print(summary(report, validation), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
