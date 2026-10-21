"""Phase 07C frozen public evaluation. Run manually in metal_ad_ead.

No optimization, threshold selection, validation recalibration, or formal F1.
Native metrics are exact CPU metrics; insufficient RAM fails explicitly.
"""

import csv
import ctypes
import gc
import hashlib
import inspect
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "phase07c_baseline256_public"
CHECKPOINT = ROOT / "experiments/phase07b_baseline256_seed42/checkpoints/final.ckpt"
EXPECTED_SHA256 = "37080cefb4619e1577a168a2000772b695ab89b20dacae890ef2ca371a3afb3a"
MIB = 1024 ** 2
FIELDS = ["sample_id", "image_path", "gt_label", "gt_mask_path", "anomalib_pred_score",
          "official_image_score", "score_absolute_difference", "anomaly_map_min", "anomaly_map_max",
          "anomaly_map_mean", "anomaly_map_shape", "native_gt_shape", "defect_pixels",
          "raw_map_path", "model_only_inference_ms", "end_to_end_ms", "warmup"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * MIB), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initial_report():
    return dict(status="NOT_RUN", timestamp=None, checkpoint_path=str(CHECKPOINT),
                expected_checkpoint_sha256=EXPECTED_SHA256, checkpoint_sha256=None,
                dataset="MVTec AD 2", category="sheet_metal", test_split="test_public",
                model="EfficientAD-S", model_size="small", input_size=[256, 256], seed=42,
                resize_policy="Default EfficientAD preprocessing; native map resize bilinear, align_corners=False",
                native_gt_shape=None, native_shape_order="height,width (PIL width,height is 4224,1056)",
                model_output_shape=None, interpolation_method="bilinear; align_corners=False",
                postprocessing="Disabled in frozen Phase 07B checkpoint; kept disabled",
                aupro_fpr_limits=[0.05, 0.30],
                metrics={"image_auroc": None, "pixel_auroc": None, "aupro_0.05": None, "aupro_0.30": None},
                predictions_saved=0, visualizations_saved=0,
                inference_timing={}, peak_inference_allocated_mib=None, peak_inference_reserved_mib=None,
                warnings=[], error_message=None, traceback=None,
                metric_memory_policy="Native-resolution exact CPU metrics; disk-backed native arrays. "
                                     "Official AUPRO still needs global CPU sorting; RAM guard rejects unsafe allocation.",
                timing_policy="First 10 samples excluded. Forward hooks time model.model only with CUDA synchronization. "
                              "End-to-end starts before dataset read and ends after prediction, before CSV/NPY/PNG writes.")


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def prepare_outputs():
    for name in ("", "raw_anomaly_maps", "visualizations", "metrics"):
        (OUT / name).mkdir(parents=True, exist_ok=True)


def summary(r):
    lines = ["=" * 40, "PHASE 07C - BASELINE PUBLIC EVALUATION", "=" * 40,
             "Dataset: MVTec AD 2 / sheet_metal", "Split: test_public",
             f"Test images: {r.get('test_count', 'N/A')}", f"Normal: {r.get('normal_count', 'N/A')}",
             f"Anomaly: {r.get('anomaly_count', 'N/A')}", "Model: EfficientAD-S", "Input: 256x256",
             f"Checkpoint SHA256: {r['checkpoint_sha256']}", "-" * 40]
    for label, key in (("Image AUROC", "image_auroc"), ("Pixel AUROC", "pixel_auroc"),
                       ("AU-PRO@0.05", "aupro_0.05"), ("AU-PRO@0.30", "aupro_0.30")):
        value = r["metrics"].get(key)
        lines.append(f"{label}: " + ("N/A" if value is None else f"{value * 100:.2f} %"))
    for label, key in (("Model-only inference", "model_only"), ("End-to-end inference", "end_to_end")):
        stats = r["inference_timing"].get(key, {})
        lines.append(f"{label}: {stats.get('mean_ms', 'N/A')} ms/image")
    lines.extend([f"FPS (model-only): {r['inference_timing'].get('model_only', {}).get('fps', 'N/A')}",
                  f"Peak inference allocated VRAM: {r['peak_inference_allocated_mib']} MiB",
                  f"Peak inference reserved VRAM: {r['peak_inference_reserved_mib']} MiB",
                  f"Raw anomaly maps saved: {r['predictions_saved']}",
                  f"Visualizations saved: {r['visualizations_saved']}", "STATUS:", r["status"]])
    if r["error_message"]:
        lines.append(r["error_message"])
    return "\n".join(lines + ["=" * 40])


def write_reports(r):
    failures = []
    for name in ("evaluation_report.json", "evaluation_report.txt"):
        try:
            if name.endswith("json"):
                write_json(OUT / name, r)
            else:
                (OUT / name).write_text(summary(r) + "\n\n" + json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            traceback.print_exc()
            failures.append(str(exc))
    if failures:
        raise OSError("; ".join(failures))


def available_ram():
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in ("total", "available", "page_total", "page_available",
                                                   "virtual_total", "virtual_available", "extended")]
    state = MemoryStatus()
    state.length = ctypes.sizeof(state)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
        raise OSError("Unable to query physical RAM for native metrics")
    return state.available


def guard_ram(pixels, bytes_per_pixel, report, name):
    needed = pixels * bytes_per_pixel + 2 * 1024 ** 3
    available = available_ram()
    report.setdefault("ram_checks", {})[name] = dict(estimated_required_bytes=needed, available_bytes=available)
    if needed > available * 0.85:
        raise MemoryError(f"{name} exact native metric requires conservatively {needed / 1024**3:.1f} GiB RAM; "
                          f"available {available / 1024**3:.1f} GiB. Predictions retained; no downsampling/approximation.")


def native_mask(row, np, Image):
    shape = tuple(json.loads(row["native_gt_shape"]))
    if int(row["gt_label"]) == 0:
        return np.zeros(shape, dtype=np.uint8)
    with Image.open(row["gt_mask_path"]) as mask:
        array = (np.asarray(mask.convert("L")) > 0).astype(np.uint8)
    if array.shape != shape or not array.any():
        raise ValueError(f"Invalid native GT: {row['gt_mask_path']}")
    return array


def visualize(rows, np, Image, report):
    # GT area is used only for display selection; never for model/metric settings.
    normals = [row for row in rows if row["gt_label"] == 0][:5]
    anomalies = sorted((row for row in rows if row["gt_label"] == 1), key=lambda row: row["defect_pixels"])
    selected = normals + [anomalies[i] for i in np.linspace(0, len(anomalies) - 1, 10, dtype=int)]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for row in selected:
        with Image.open(row["image_path"]) as image:
            width, height = image.size
            size = (min(width, 1200), max(1, round(height * min(width, 1200) / width)))
            rgb = np.asarray(image.convert("RGB").resize(size, Image.Resampling.BILINEAR))
        mask = native_mask(row, np, Image)
        mask = np.asarray(Image.fromarray(mask * 255).resize(size, Image.Resampling.NEAREST))
        raw = np.load(row["raw_map_path"])
        display = np.asarray(Image.fromarray(raw).resize(size, Image.Resampling.BILINEAR))
        display = (display - display.min()) / max(float(display.max() - display.min()), 1e-12)
        fig, axes = plt.subplots(4, 1, figsize=(12, 12), constrained_layout=True)
        try:
            for axis, title in zip(axes, ("Original", "GT mask", "Heatmap (display normalization)", "Overlay")):
                axis.set_title(title)
                axis.axis("off")
            axes[0].imshow(rgb)
            axes[1].imshow(mask, cmap="gray", vmin=0, vmax=255)
            axes[2].imshow(display, cmap="jet", vmin=0, vmax=1)
            axes[3].imshow(rgb)
            axes[3].imshow(display, cmap="jet", vmin=0, vmax=1, alpha=0.45)
            fig.savefig(OUT / "visualizations" / f"{row['sample_id']}.png", dpi=120)
        finally:
            plt.close(fig)
        report["visualizations_saved"] += 1


def main():
    report = initial_report()
    report["timestamp"] = datetime.now().astimezone().isoformat()
    stage = "FAIL_OTHER"
    rows, handles = [], []
    torch = None
    memory_ready = False
    csv_file = None
    start = time.perf_counter()
    code = 1
    try:
        os.chdir(ROOT)
        prepare_outputs()
        stage = "FAIL_CHECKPOINT_HASH"
        if not CHECKPOINT.is_file():
            raise FileNotFoundError(CHECKPOINT)
        report["checkpoint_sha256"] = sha256(CHECKPOINT)
        if report["checkpoint_sha256"] != EXPECTED_SHA256:
            raise ValueError("Frozen checkpoint SHA256 mismatch; evaluation stopped.")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        import numpy as np
        import torch
        import torchvision
        from PIL import Image
        from importlib.metadata import version
        from dataclasses import fields
        from lightning.pytorch import Callback, seed_everything
        from torch.utils.data import Dataset, DataLoader
        from anomalib.data import MVTecAD2, ImageBatch
        from anomalib.models import EfficientAd
        from anomalib.engine import Engine
        from anomalib.metrics import AUPRO
        from sklearn.metrics import roc_auc_score

        report.update(python_version=platform.python_version(), torch_version=str(torch.__version__),
                      torchvision_version=torchvision.__version__, cuda_runtime=torch.version.cuda,
                      anomalib_version=version("anomalib"),
                      aupro_signature=str(inspect.signature(AUPRO)),
                      aupro_implementation_signature=str(inspect.signature(AUPRO.__mro__[2])),
                      predict_batch_fields=[field.name for field in fields(ImageBatch)])
        seed_everything(42, workers=True)
        stage = "FAIL_INFERENCE"
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU 0 required")
        torch.cuda.set_device(0)
        props = torch.cuda.get_device_properties(0)
        report.update(gpu_name=props.name, gpu_total_memory_mib=props.total_memory / MIB)
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version", "--format=csv"],
                                    capture_output=True, text=True, timeout=20,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False)
            report["nvidia_driver_info"] = result.stdout.strip() or result.stderr.strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            report["warnings"].append(f"Driver query unavailable: {exc}")
        stage = "FAIL_DATA"
        if not (ROOT / "datasets/MVTec_AD_2/sheet_metal").is_dir():
            raise FileNotFoundError("Existing dataset required; no dataset download allowed")
        dm = MVTecAD2(root="./datasets/MVTec_AD_2", category="sheet_metal", train_batch_size=1,
                     eval_batch_size=1, num_workers=0, test_type="public")
        dm.setup()
        for name, data, expected in (("train_count", dm.train_data, 137), ("validation_count", dm.val_data, 19),
                                     ("test_count", dm.test_public_data, 114)):
            report[name] = len(data)
            if len(data) != expected:
                raise ValueError(f"{name}: expected {expected}, got {len(data)}")
        loader = dm.test_dataloader()
        samples = loader.dataset.samples
        labels = samples.label_index.astype(int).tolist()
        report.update(normal_count=labels.count(0), anomaly_count=labels.count(1))
        if len(loader) != 114 or labels.count(0) != 24 or labels.count(1) != 90:
            raise ValueError("Public loader/GT counts do not match 114/24/90")
        metadata = {}
        for sample in samples.itertuples():
            path = str(Path(sample.image_path).resolve())
            with Image.open(path) as image:
                width, height = image.size
            if (width, height) != (4224, 1056):
                raise ValueError(f"Unexpected native image dimensions: {path}: {(width, height)}")
            mask_path = str(Path(sample.mask_path).resolve()) if int(sample.label_index) else ""
            if mask_path:
                with Image.open(mask_path) as mask:
                    if mask.size != (width, height) or not np.asarray(mask).any():
                        raise ValueError(f"Invalid anomalous GT mask: {mask_path}")
            metadata[path] = (int(sample.label_index), mask_path, [height, width])
        if len(metadata) != 114:
            raise ValueError("Duplicate public image paths")
        report["native_gt_shape"] = [1056, 4224]
        stage = "FAIL_CHECKPOINT_LOAD"
        saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        frozen = saved.get("phase07b", {})
        if saved.get("global_step") != 70000 or not frozen.get("final_validation_completed"):
            raise ValueError("Checkpoint lacks 70000-step/final-validation provenance")
        if frozen.get("config", {}).get("post_processor") is not False:
            raise ValueError("Frozen postprocessing does not match Phase 07B")
        del saved
        model = EfficientAd(model_size="small", imagenet_dir="./datasets/imagenette",
                            evaluator=False, visualizer=False, post_processor=False)
        model.requires_grad_(False)
        model.eval()
        timing = {}

        class TimedDataset(Dataset):
            def __len__(self):
                return len(loader.dataset)

            def __getitem__(self, index):
                timing["e2e_start"] = time.perf_counter()
                return loader.dataset[index]

        def forward_start(module, args):
            torch.cuda.synchronize()
            timing["forward_start"] = time.perf_counter()

        def forward_end(module, args, output):
            torch.cuda.synchronize()
            timing["forward_ms"] = (time.perf_counter() - timing["forward_start"]) * 1000

        handles.extend([model.model.register_forward_pre_hook(forward_start),
                        model.model.register_forward_hook(forward_end)])
        csv_file = (OUT / "predictions.csv").open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        writer.writeheader()

        class SavePrediction(Callback):
            def on_predict_start(self, trainer, pl_module):
                nonlocal stage
                stage = "FAIL_INFERENCE"  # official checkpoint restore succeeded

            def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
                torch.cuda.synchronize()
                e2e_ms = (time.perf_counter() - timing["e2e_start"]) * 1000
                path = str(Path(outputs.image_path[0]).resolve())
                label, mask_path, shape = metadata[path]
                raw = outputs.anomaly_map.detach().float().cpu().squeeze().numpy().copy()
                if raw.shape != (256, 256) or not np.isfinite(raw).all():
                    raise ValueError(f"Invalid model map: shape={raw.shape}, path={path}")
                pred = None if outputs.pred_score is None else float(outputs.pred_score.detach().cpu().item())
                if pred is not None and not math.isfinite(pred):
                    raise ValueError("Nonfinite predicted image score")
                sample_id = f"{batch_idx:03d}_{Path(path).stem}"
                raw_path = OUT / "raw_anomaly_maps" / f"{sample_id}.npy"
                np.save(raw_path, raw.astype(np.float32, copy=False), allow_pickle=False)
                row = dict(sample_id=sample_id, image_path=path, gt_label=label, gt_mask_path=mask_path,
                           anomalib_pred_score=pred, official_image_score=float(raw.max()),
                           score_absolute_difference=None if pred is None else abs(pred - float(raw.max())),
                           anomaly_map_min=float(raw.min()), anomaly_map_max=float(raw.max()),
                           anomaly_map_mean=float(raw.mean()), anomaly_map_shape=json.dumps(list(raw.shape)),
                           native_gt_shape=json.dumps(shape), raw_map_path=str(raw_path),
                           model_only_inference_ms=timing["forward_ms"], end_to_end_ms=e2e_ms, warmup=batch_idx < 10)
                row["defect_pixels"] = int(native_mask(row, np, Image).sum())
                writer.writerow(row)
                csv_file.flush()
                rows.append(row)
                report["predictions_saved"] = len(rows)
                report["model_output_shape"] = list(raw.shape)
                if len(rows) % 10 == 0:
                    print(f"Saved {len(rows)}/114 public predictions", flush=True)

        timed_loader = DataLoader(TimedDataset(), batch_size=1, shuffle=False, num_workers=0,
                                 collate_fn=loader.dataset.collate_fn, pin_memory=True)
        engine = Engine(accelerator="gpu", devices=1, precision="32-true", logger=False,
                        barebones=True, enable_checkpointing=False, default_root_dir=str(OUT),
                        num_sanity_val_steps=0, callbacks=[SavePrediction()])
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        memory_ready = True
        engine.predict(model=model, dataloaders=timed_loader, ckpt_path=str(CHECKPOINT), return_predictions=False)
        report["peak_inference_allocated_mib"] = torch.cuda.max_memory_allocated() / MIB
        report["peak_inference_reserved_mib"] = torch.cuda.max_memory_reserved() / MIB
        if len(rows) != 114 or len({row["image_path"] for row in rows}) != 114:
            raise ValueError("Incomplete or duplicate predictions")
        csv_file.close()
        csv_file = None
        for name, column in (("model_only", "model_only_inference_ms"), ("end_to_end", "end_to_end_ms")):
            values = np.array([row[column] for row in rows[10:]], dtype=np.float64)
            report["inference_timing"][name] = dict(count=len(values), mean_ms=float(values.mean()),
                                                   median_ms=float(np.median(values)), std_ms=float(values.std()),
                                                   fps=float(1000 / values.mean()))
        report["max_score_absolute_difference"] = max((row["score_absolute_difference"] or 0) for row in rows)
        for handle in handles:
            handle.remove()
        handles.clear()
        model.cpu()
        torch.cuda.empty_cache()
        stage = "FAIL_METRIC"
        metrics = report["metrics"]
        metrics.update(test_count=114, normal_count=24, anomaly_count=90,
                       image_auroc=float(roc_auc_score([row["gt_label"] for row in rows],
                                                       [row["official_image_score"] for row in rows])))

        def persist_metrics():
            metrics["percent"] = {key: value * 100 for key, value in metrics.items()
                                  if key in ("image_auroc", "pixel_auroc", "aupro_0.05", "aupro_0.30")
                                  and value is not None}
            write_json(OUT / "metrics/public_metrics.json", metrics)

        persist_metrics()
        # Visualization errors are deferred so they cannot discard formal metrics.
        visual_error = None
        try:
            visualize(rows, np, Image, report)
        except Exception as exc:
            traceback.print_exc()
            visual_error = str(exc)
            report["warnings"].append(f"Visualization failed: {exc}")
        shape = (114, 1056, 4224)
        pixels = math.prod(shape)
        if shutil.disk_usage(OUT).free < pixels * 5 + 1024 ** 3:
            raise OSError("Insufficient disk space for native CPU metric arrays")
        scores = np.lib.format.open_memmap(OUT / "metrics/native_scores.npy", mode="w+", dtype=np.float32, shape=shape)
        targets = np.lib.format.open_memmap(OUT / "metrics/native_targets.npy", mode="w+", dtype=np.uint8, shape=shape)
        for i, row in enumerate(rows):
            raw = torch.from_numpy(np.load(row["raw_map_path"]))[None, None]
            scores[i] = torch.nn.functional.interpolate(raw, size=shape[1:], mode="bilinear",
                                                        align_corners=False)[0, 0].numpy()
            targets[i] = native_mask(row, np, Image)
        scores.flush()
        targets.flush()
        guard_ram(pixels, 40, report, "Pixel AUROC")
        metrics["pixel_auroc"] = float(roc_auc_score(targets.reshape(-1), scores.reshape(-1)))
        persist_metrics()
        gc.collect()
        stage = "FAIL_AUPRO"
        guard_ram(pixels, 96, report, "AUPRO")
        # Explicit limits, no default .30, no threshold search or GT-dependent settings.
        for metric, key in ((AUPRO(fields=["anomaly_map", "gt_mask"], fpr_limit=0.05), "aupro_0.05"),
                            (AUPRO(fields=["anomaly_map", "gt_mask"], fpr_limit=0.30), "aupro_0.30")):
            metric = metric.cpu()
            metric.update(SimpleNamespace(anomaly_map=torch.from_numpy(scores), gt_mask=torch.from_numpy(targets)))
            value = float(metric.compute().item())
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{key} outside [0,1]: {value}")
            metrics[key] = value
            metric.reset()
            gc.collect()
            persist_metrics()
        del scores, targets
        stage = "FAIL_CHECKPOINT_HASH"
        if sha256(CHECKPOINT) != EXPECTED_SHA256:
            raise ValueError("Checkpoint changed during evaluation")
        stage = "FAIL_VISUALIZATION"
        if visual_error is not None:
            raise RuntimeError(visual_error)
        report["status"] = "PASS"
        code = 0
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status=stage, error_message=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        if csv_file is not None:
            csv_file.close()
        for handle in handles:
            handle.remove()
        if memory_ready and report["peak_inference_allocated_mib"] is None:
            try:
                report["peak_inference_allocated_mib"] = torch.cuda.max_memory_allocated() / MIB
                report["peak_inference_reserved_mib"] = torch.cuda.max_memory_reserved() / MIB
            except Exception as exc:
                report["warnings"].append(f"VRAM stats unavailable: {exc}")
        report["elapsed_seconds"] = time.perf_counter() - start
        try:
            prepare_outputs()
            write_reports(report)
        except Exception as exc:
            traceback.print_exc()
            report.update(status="FAIL_OTHER", error_message=f"Report write failed: {exc}")
            code = 1
        print(summary(report), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
