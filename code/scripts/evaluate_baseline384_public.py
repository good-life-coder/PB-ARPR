"""Phase 08C frozen public evaluation. Run manually in metal_ad_ead.

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.evaluation import exact_aupro as exact
from scripts.calibrate_resolution import official_preprocessor
from scripts.analyze_b0_failures import size_group

B0 = ROOT / "experiments/phase07c_baseline256_public"
D0 = ROOT / "experiments/phase07d_b0_failure_analysis"
R1 = ROOT / "experiments/phase07c_r1_aupro_exact"
OUT = ROOT / "experiments" / "phase08c_baseline384_public"
CHECKPOINT = ROOT / "experiments/phase08b_baseline384_seed42/checkpoints/final.ckpt"
EXPECTED_SHA256 = "d8d2c8fb961c5d61dc936156286b16ce9c097ac28350c66b3cabae58c5134cec"
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
                model="EfficientAD-S", model_size="small", input_size=[384, 384], seed=42,
                resize_policy="Default EfficientAD preprocessing; native map resize bilinear, align_corners=False",
                native_gt_shape=None, native_shape_order="height,width (PIL width,height is 4224,1056)",
                model_output_shape=None, interpolation_method="bilinear; align_corners=False",
                image_score_policy="max of saved continuous model-output map, before native interpolation; identical to B0",
                approximation=False,
                postprocessing="Disabled in frozen Phase 08B checkpoint; kept disabled",
                aupro_fpr_limits=[0.05, 0.30],
                metrics={"image_auroc": None, "pixel_auroc": None, "aupro_0.05": None, "aupro_0.30": None},
                predictions_saved=0, visualizations_saved=0,
                inference_timing={}, peak_inference_allocated_mib=None, peak_inference_reserved_mib=None,
                warnings=[], error_message=None, traceback=None,
                metric_memory_policy="Native-resolution CPU Pixel AUROC; shared R1 exact rank AU-PRO with background memmap sorting.",
                timing_policy="First 10 samples excluded. Forward hooks time model.model only with CUDA synchronization. "
                              "End-to-end starts before dataset read and ends after prediction, before CSV/NPY/PNG writes.")


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def prepare_outputs():
    for name in ("", "raw_anomaly_maps", "visualizations", "metrics"):
        (OUT / name).mkdir(parents=True, exist_ok=True)


def summary(r):
    lines = ["=" * 40, "PHASE 08C - 384 PUBLIC EVALUATION", "=" * 40,
             "Dataset: MVTec AD 2 / sheet_metal", "Split: test_public",
             f"Test images: {r.get('test_count', 'N/A')}", f"Normal: {r.get('normal_count', 'N/A')}",
             f"Anomaly: {r.get('anomaly_count', 'N/A')}", "Model: EfficientAD-S", "Input: 384x384",
             f"Checkpoint SHA256: {r['checkpoint_sha256']}", "-" * 40]
    for label, key in (("Image AUROC", "image_auroc"), ("Pixel AUROC", "pixel_auroc"),
                       ("AU-PRO@0.05", "aupro_0.05"), ("AU-PRO@0.30", "aupro_0.30")):
        value = r["metrics"].get(key)
        lines.append(f"{label}: " + ("N/A" if value is None else f"{value * 100:.2f} %"))
    for row in r.get("comparison", []):
        lines.append(f"{row['Metric']}: B0={row['B0_256']}, B1={row['B1_384']}, gain={row['Absolute_Gain']}")
    lines.append("Decision facts only: " + json.dumps(r.get("decision_facts", {})))
    lines.append("Approximation: False; no automatic 512 training decision.")
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


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_references(metadata):
    """Freeze membership and visualization choices from B0, before B1 inference."""
    analysis = json.loads((D0 / "failure_analysis.json").read_text(encoding="utf-8"))
    recovery = json.loads((R1 / "aupro_report.json").read_text(encoding="utf-8"))
    validation = json.loads((R1 / "validation_report.json").read_text(encoding="utf-8"))
    if analysis["status"] != "PASS" or recovery["status"] != "PASS" or validation["status"] != "PASS":
        raise ValueError("B0 analysis and exact AU-PRO validation must be frozen PASS")
    if sha256(B0 / "predictions.csv") != recovery["input_csv_sha256"]:
        raise ValueError("B0 predictions changed since R1 validation")
    predictions = read_csv(B0 / "predictions.csv")
    samples = read_csv(D0 / "sample_analysis.csv")
    groups = read_csv(D0 / "size_group_analysis.csv")
    by_path = {str(Path(row["image_path"]).resolve()): row for row in predictions}
    if len(by_path) != 114 or set(by_path) != set(metadata):
        raise ValueError("B0/B1 public sample paths differ")
    for path, row in by_path.items():
        label, mask, shape = metadata[path]
        if int(row["gt_label"]) != label or (label and str(Path(row["gt_mask_path"]).resolve()) != mask):
            raise ValueError(f"B0/B1 label or mask mapping differs: {path}")
    frozen_hashes = {str(Path(item["path"]).resolve()): item["sha256"] for item in recovery["input_raw_map_hashes"]}
    for row in predictions:
        path = Path(row["raw_map_path"]).resolve()
        if sha256(path) != frozen_hashes.get(str(path)):
            raise ValueError(f"B0 raw map changed: {path}")
    if len(samples) != 90 or len({row["sample_id"] for row in samples}) != 90:
        raise ValueError("B0 analysis must contain 90 unique anomalies")
    memberships = {row["sample_id"]: row for row in samples}
    anomaly_ids = {row["sample_id"] for row in predictions if int(row["gt_label"]) == 1}
    if set(memberships) != anomaly_ids:
        raise ValueError("B0 group membership does not cover all anomaly samples")
    if sum(row["size_group"] == "A" for row in samples) != 12:
        raise ValueError("Frozen tiny group must contain 12 samples")
    by_id = {row["sample_id"]: row for row in predictions}
    chosen = {}
    for name in ("worst_10", "smallest_10", "highest_normal_response_10"):
        ids = analysis["gallery_selection"][name]
        if len(ids) != 10 or len(set(ids)) != 10 or not set(ids) <= set(by_id):
            raise ValueError(f"Invalid fixed B0 gallery list: {name}")
        chosen[name] = ids
    return dict(predictions=by_id, memberships=memberships, groups=groups, galleries=chosen,
                source_hashes={str(p): sha256(p) for p in (B0 / "predictions.csv", D0 / "sample_analysis.csv",
                                                         D0 / "failure_analysis.json", D0 / "size_group_analysis.csv")})


def finish_comparison(rows, report, references, np, torch, cv2, Image):
    per_image = []
    for row in rows:
        if not row["gt_label"]:
            continue
        frozen = references["memberships"][row["sample_id"]]
        mask = native_mask(row, np, Image)
        count = int(mask.sum())
        group = size_group(count / mask.size)
        if count != int(frozen["defect_pixel_count"]) or group != frozen["size_group"]:
            raise ValueError("B1 group membership differs from frozen B0")
        native_row = dict(row, native_shape=tuple(mask.shape))
        score = exact.get_score(native_row, torch, np)
        labels, sizes = exact.components(mask, cv2, np)
        fg = labels > 0
        background = score[~fg]
        background.sort(kind="quicksort")
        weights = 1.0 / ((len(sizes)-1) * sizes[labels[fg]].astype(np.float64))
        values, weight_sum = exact.integrate(background, score[fg], weights, np)
        if abs(weight_sum-1) >= 1e-6 or not all(math.isfinite(v) and -1e-12 <= v <= 1+1e-12 for v in values.values()):
            raise ValueError("Invalid per-image exact integral")
        per_image.append(dict(sample_id=row["sample_id"], image_path=row["image_path"], group=group,
                              defect_ratio=count/mask.size, aupro_005=values[0.05]))
    groups = []
    for frozen in references["groups"]:
        group = frozen["group"]
        values = np.array([row["aupro_005"] for row in per_image if row["group"] == group])
        if len(values) != int(frozen["sample_count"]):
            raise ValueError("B0/B1 size-group counts differ")
        groups.append(dict(group=group, range=frozen["range"], n=len(values),
                           mean_aupro005=float(values.mean()) if len(values) else None,
                           median_aupro005=float(np.median(values)) if len(values) else None,
                           std_aupro005=float(values.std()) if len(values) else None))
    report["size_groups"] = groups
    report["fixed_gallery_selection"] = references["galleries"]
    for path, content in ((OUT / "size_group_analysis.csv", groups), (OUT / "metrics/per_image_aupro.csv", per_image)):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(content[0]))
            writer.writeheader()
            writer.writerows(content)
    comparison = []

    def add(name, baseline, current, percentage=False):
        gain = current-baseline if current is not None else None
        comparison.append(dict(Metric=name, B0_256=baseline, B1_384=current,
                               Absolute_Gain=(f"{gain:+.6f} pp" if percentage and gain is not None else gain),
                               Relative_Change=None if baseline == 0 or gain is None else gain/baseline))
    for name, key, baseline in (("Image AUROC", "image_auroc", 61.57), ("Pixel AUROC", "pixel_auroc", 79.54),
                                 ("AU-PRO@0.05", "aupro_0.05", 6.41), ("AU-PRO@0.30", "aupro_0.30", 22.33)):
        add(name, baseline, report["metrics"][key]*100, True)
    b0_groups = dict(A=0.00, B=7.23, C=10.11, D=45.57, E=29.49)
    for row in groups:
        add(f"{row['range']} mean AU-PRO@0.05", b0_groups[row["group"]],
            None if row["mean_aupro005"] is None else row["mean_aupro005"]*100, True)
    inference = report["inference_timing"]["model_only"]
    for name, baseline, current in (("Model-only inference ms", 43.1369, inference["mean_ms"]),
                                    ("FPS", 23.1820, inference["fps"]), ("Training time hours", 4.2807, 8.0460),
                                    ("Training peak VRAM MiB", 1185.2217, 1550.7520),
                                    ("Inference peak VRAM MiB", 873.2402, report["peak_inference_allocated_mib"])):
        add(name, baseline, current)
    report["comparison"] = comparison
    report["comparison_units"] = "Score metrics in percent; gains in percentage points. Relative_Change is a fraction; undefined for zero B0."
    by_group = {row["group"]: row["mean_aupro005"] for row in groups}
    report["decision_facts"] = dict(overall_aupro005_gain_pp=report["metrics"]["aupro_0.05"]*100-6.41,
                                    tiny_group_gain_pp=by_group["A"]*100,
                                    groupB_gain_pp=by_group["B"]*100-7.23,
                                    groupC_gain_pp=by_group["C"]*100-10.11,
                                    training_time_ratio=8.0460/4.2807,
                                    inference_time_ratio=inference["mean_ms"]/43.1369,
                                    training_vram_ratio=1550.7520/1185.2217)
    with (OUT / "metrics/B0_vs_B1.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    write_reports(report)  # Keep metric/group results if a gallery fails.
    comparison_gallery(rows, references, report, np, torch, Image)


def comparison_gallery(rows, references, report, np, torch, Image):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by_id = {row["sample_id"]: row for row in rows}
    report["comparison_visualizations_saved"] = 0
    for name, ids in references["galleries"].items():
        folder = OUT / "visualizations/comparison" / name
        folder.mkdir(parents=True, exist_ok=True)
        for sample_id in ids:
            row = by_id[sample_id]
            raw0 = np.load(references["predictions"][sample_id]["raw_map_path"], allow_pickle=False)
            raw1 = np.load(row["raw_map_path"], allow_pickle=False)
            shape = tuple(json.loads(row["native_gt_shape"]))
            displays = []
            for raw in (raw0, raw1):
                native = torch.nn.functional.interpolate(torch.from_numpy(raw)[None,None], size=shape,
                                                         mode="bilinear", align_corners=False)[0,0].numpy()
                displays.append(np.asarray(Image.fromarray(native).resize((1200,300), Image.Resampling.BILINEAR)))
            lo, hi = min(x.min() for x in displays), max(x.max() for x in displays)
            with Image.open(row["image_path"]) as image:
                rgb = np.asarray(image.convert("RGB").resize((1200,300), Image.Resampling.BILINEAR))
            mask = np.asarray(Image.fromarray(native_mask(row,np,Image)*255).resize((1200,300),Image.Resampling.NEAREST))
            fig, axes = plt.subplots(2,2,figsize=(16,6),constrained_layout=True)
            try:
                for ax,title in zip(axes.flat,("Original","GT","B0 256 heatmap","B1 384 heatmap")):
                    ax.set_title(title)
                    ax.axis("off")
                axes[0,0].imshow(rgb)
                axes[0,1].imshow(mask,cmap="gray",vmin=0,vmax=255)
                for ax,display in zip(axes[1],displays):
                    ax.imshow(display,cmap="jet",vmin=lo,vmax=max(hi,lo+1e-12))
                fig.suptitle(sample_id+" | shared display-only color scale")
                fig.savefig(folder/f"{sample_id}.png",dpi=150)
            finally:
                plt.close(fig)
            report["comparison_visualizations_saved"] += 1


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
        import cv2
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
        references = load_references(metadata)
        report["b0_mapping_check"] = "PASS"
        stage = "FAIL_CHECKPOINT_LOAD"
        saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        frozen = saved.get("phase08b", {})
        if saved.get("global_step") != 70000 or not frozen.get("final_validation_completed"):
            raise ValueError("Checkpoint lacks 70000-step/final-validation provenance")
        if frozen.get("config", {}).get("post_processor") is not False:
            raise ValueError("Frozen postprocessing does not match Phase 08B")
        if frozen.get("config", {}).get("input_size") != [384, 384]:
            raise ValueError("Checkpoint preprocessing resolution is not 384")
        preprocessor, preprocessing = official_preprocessor(EfficientAd, 384)
        if frozen.get("config", {}).get("preprocessor_configuration", {}).get("selected_transform") != preprocessing["selected_transform"]:
            raise ValueError("Preprocessing differs from frozen Phase 08B")
        report["preprocessor"] = preprocessing
        print("Input: 384x384", flush=True)
        print("B0 preprocessing:", preprocessing["default_transform"], flush=True)
        print("384 preprocessing:", preprocessing["selected_transform"], flush=True)
        print("Only resize differs:", preprocessing["only_resize_changed"], flush=True)
        del saved
        model = EfficientAd(model_size="small", imagenet_dir="./datasets/imagenette",
                            pre_processor=preprocessor, evaluator=False, visualizer=False, post_processor=False)
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
                if raw.shape != (384, 384) or not np.isfinite(raw).all():
                    raise ValueError(f"Invalid model map: shape={raw.shape}, path={path}")
                pred = None if outputs.pred_score is None else float(outputs.pred_score.detach().cpu().item())
                if pred is not None and not math.isfinite(pred):
                    raise ValueError("Nonfinite predicted image score")
                sample_id = f"{batch_idx:03d}_{Path(path).stem}"
                if references["predictions"].get(sample_id, {}).get("image_path") != path:
                    raise ValueError("Prediction ordering/sample ID differs from B0")
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
        del scores, targets
        gc.collect()
        exact_report = {}
        native_rows = [dict(row, native_shape=tuple(json.loads(row["native_gt_shape"]))) for row in rows]
        exact.native(native_rows, exact_report, torch, cv2, np, Image, temp_dir=OUT / "metrics/exact_temp")
        report["exact_aupro"] = exact_report
        report["approximation"] = False
        metrics.update({key: exact_report[key] for key in ("aupro_0.05", "aupro_0.30")})
        persist_metrics()
        stage = "FAIL_ANALYSIS"
        finish_comparison(rows, report, references, np, torch, cv2, Image)
        for path, expected in references["source_hashes"].items():
            if sha256(Path(path)) != expected:
                raise ValueError(f"B0 reference changed: {path}")
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

