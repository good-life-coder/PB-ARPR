"""Phase 07D: descriptive failure analysis of immutable saved B0 predictions.

Run manually. No model construction, inference, threshold selection or training.
Per-image/region metrics are diagnostics, never replacements for frozen B0 metrics.
"""

import csv
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

try:
    from . import recover_aupro_native_exact as exact
except ImportError:
    import recover_aupro_native_exact as exact

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/phase07d_b0_failure_analysis"
R1 = ROOT / "experiments/phase07c_r1_aupro_exact"
BASELINE = {
    "name": "B0", "model": "EfficientAD-S", "dataset": "MVTec AD 2", "category": "sheet_metal",
    "split": "test_public", "input_size": [256, 256], "training_steps": 70000, "seed": 42,
    "image_auroc_percent": 61.57, "pixel_auroc_percent": 79.54,
    "aupro_005_percent": 6.41, "aupro_030_percent": 22.33,
    "training_time_hours": 4.2807, "training_peak_allocated_vram_mib": 1185.2217,
    "model_only_inference_ms": 43.1369, "fps": 23.1820,
    "inference_peak_allocated_vram_mib": 873.2402,
    "checkpoint_sha256": "37080cefb4619e1577a168a2000772b695ab89b20dacae890ef2ca371a3afb3a",
}
GROUPS = [("A", "<0.05%"), ("B", "0.05-0.1%"), ("C", "0.1-0.5%"), ("D", "0.5-1%"), ("E", ">=1%")]
SAMPLE_FIELDS = ["sample_id", "image_path", "defect_pixel_count", "total_pixel_count", "defect_ratio",
                 "connected_component_count", "size_group", "global_max", "gt_mean", "gt_max", "gt_p95",
                 "background_mean", "background_p99", "background_p999", "pixel_auroc", "aupro_005", "aupro_030"]
NORMAL_FIELDS = ["rank", "sample_id", "image_path", "max_score", "mean_score", "p99_score", "p999_score"]
REGION_FIELDS = ["sample_id", "image_path", "region_id", "region_pixel_count", "region_ratio_to_image",
                 "bbox_x", "bbox_y", "bbox_width", "bbox_height", "bbox_aspect_ratio",
                 "distance_to_nearest_image_border", "border_distance_normalized", "score_mean", "score_max",
                 "score_p95", "region_aupro_005", "region_aupro_030", "area_quantile_group"]
GROUP_METRICS = ["defect_ratio", "pixel_auroc", "aupro_005", "aupro_030", "gt_mean", "background_p999"]
GROUP_FIELDS = ["group", "range", "sample_count"] + [f"{name}_{stat}" for name in GROUP_METRICS
                                                        for stat in ("mean", "median", "std", "min", "max")]


def initial_report():
    return dict(status="NOT_RUN", baseline=BASELINE, error_message=None, warnings=[],
                source="phase07c_baseline256_public", exact_algorithm_source="phase07c_r1_aupro_exact",
                hypotheses={}, correlations={}, notes=[
                    "Per-image and per-region AU-PRO are diagnostic and do not redefine the pooled formal B0 result.",
                    "Normal images are ranked by response; without a frozen segmentation threshold, these are not confirmed false positives.",
                    "Spearman associations are descriptive, not causal; exploratory p-values are unadjusted.",
                    "Population standard deviation (ddof=0). Empty groups report null statistics.",
                    "Region quantiles are for analysis only; ties stay in the same lower inclusive quantile group.",
                    "Region distance is in native pixels; region AU-PRO uses that image's background distribution.",
                    "Region observations within an image are dependent; H4 can be at most PARTIALLY_SUPPORTED by this descriptive analysis.",
                ], decision_rules={
                    "H1": "Compare fixed A vs pooled D/E; fewer than 3 in either is INCONCLUSIVE. "
                          "Positive mean and median differences plus two-sided Mann-Whitney p<0.05: SUPPORTED; "
                          "both positive without significance: PARTIALLY_SUPPORTED; neither positive: NOT_SUPPORTED; otherwise INCONCLUSIVE.",
                    "H2": "Positive Spearman rho and p<0.05: SUPPORTED; positive rho otherwise: PARTIALLY_SUPPORTED; "
                          "nonpositive rho: NOT_SUPPORTED; missing/constant data: INCONCLUSIVE.",
                    "H3": "No threshold-based FP rate claimed. PARTIALLY_SUPPORTED only if at least one normal maximum "
                          "exceeds anomalous-image score Q95 and normal-over-anomaly pairwise rank overlap >=0.25; otherwise INCONCLUSIVE.",
                    "H4": "Positive rank association of border distance and region PRO, also positive after controlling "
                          "region area ranks: PARTIALLY_SUPPORTED; both nonpositive: NOT_SUPPORTED; otherwise INCONCLUSIVE. "
                          "No claim of a border effect independent of image-level confounding.",
                })


def prepare_outputs():
    for folder in (OUT, OUT / "figures", OUT / "galleries"):
        folder.mkdir(parents=True, exist_ok=True)


def write_csv(name, rows, fields):
    with (OUT / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def size_group(ratio):
    for upper, group in ((0.0005, "A"), (0.001, "B"), (0.005, "C"), (0.01, "D")):
        if ratio < upper:
            return group
    return "E"


def mapping_check(np, Image, report):
    previous = json.loads((R1 / "aupro_report.json").read_text(encoding="utf-8"))
    validation = json.loads((R1 / "validation_report.json").read_text(encoding="utf-8"))
    if previous["status"] != "PASS" or validation["status"] != "PASS":
        raise ValueError("Phase 07C-R1 native exact result and lowres validation must both be PASS")
    inputs = {}
    rows = exact.read_inputs(np, Image, inputs)
    if inputs["input_csv_sha256"] != previous["input_csv_sha256"]:
        raise ValueError("Predictions CSV differs from validated Phase 07C-R1 input")
    old_hashes = {item["path"]: item["sha256"] for item in previous["input_raw_map_hashes"]}
    if {item["path"]: item["sha256"] for item in inputs["input_raw_map_hashes"]} != old_hashes:
        raise ValueError("Raw maps differ from validated Phase 07C-R1 input")
    if len({row["sample_id"] for row in rows}) != 114:
        raise ValueError("Duplicate sample IDs")
    report.update(input_integrity=inputs, input_mapping="PASS", anomaly_samples=90, normal_samples=24,
                  raw_map_shape=inputs["raw_map_shape"], native_resolution=inputs["native_resolution"],
                  exact_script_sha256=exact.digest(Path(exact.__file__)),
                  exact_source_validation=str(R1 / "validation_report.json"))
    return rows


def checked_integral(background, scores, weights, np):
    values, total = exact.integrate(background, scores, weights, np)
    if abs(total - 1) >= 1e-6 or not all(math.isfinite(v) and -1e-12 <= v <= 1+1e-12 for v in values.values()):
        raise ValueError(f"Invalid exact diagnostic integration: {values}, weight_sum={total}")
    return {key: min(1.0, max(0.0, value)) for key, value in values.items()}


def analyze_sample(row, np, torch, cv2, Image, roc_auc_score):
    score = exact.get_score(row, torch, np)
    mask = exact.get_mask(row, Image, np)
    path = row["image_path"]
    if not int(row["gt_label"]):
        return None, dict(sample_id=row["sample_id"], image_path=path,
                          max_score=float(score.max()), mean_score=float(score.mean(dtype=np.float64)),
                          p99_score=float(np.percentile(score, 99)), p999_score=float(np.percentile(score, 99.9))), []
    labels, sizes = exact.components(mask, cv2, np)
    foreground = labels > 0
    positive = score[foreground]
    background = score[~foreground]
    background.sort(kind="quicksort")
    k = len(sizes)-1
    weights = 1.0 / (k * sizes[labels[foreground]].astype(np.float64))
    pro = checked_integral(background, positive, weights, np)
    count, total = int(foreground.sum()), mask.size
    result = dict(sample_id=row["sample_id"], image_path=path, defect_pixel_count=count,
                  total_pixel_count=total, defect_ratio=count / total, connected_component_count=k,
                  size_group=size_group(count / total), global_max=float(score.max()),
                  gt_mean=float(positive.mean(dtype=np.float64)), gt_max=float(positive.max()),
                  gt_p95=float(np.percentile(positive, 95)), background_mean=float(background.mean(dtype=np.float64)),
                  background_p99=float(np.percentile(background, 99)), background_p999=float(np.percentile(background, 99.9)),
                  pixel_auroc=float(roc_auc_score(mask.reshape(-1), score.reshape(-1))),
                  aupro_005=pro[0.05], aupro_030=pro[0.30])
    region_rows = []
    height, width = mask.shape
    # One image at a time; never build a dataset-sized native tensor.
    for region_id in range(1, k+1):
        y, x = np.where(labels == region_id)
        values = score[y, x]
        w, h = int(x.max()-x.min()+1), int(y.max()-y.min()+1)
        distance = int(min(x.min(), y.min(), width-1-x.max(), height-1-y.max()))
        region_pro = checked_integral(background, values, np.full(len(values), 1.0 / len(values)), np)
        region_rows.append(dict(sample_id=row["sample_id"], image_path=path, region_id=region_id,
                                region_pixel_count=len(values), region_ratio_to_image=len(values)/total,
                                bbox_x=int(x.min()), bbox_y=int(y.min()), bbox_width=w, bbox_height=h,
                                bbox_aspect_ratio=w/h, distance_to_nearest_image_border=distance,
                                border_distance_normalized=distance/min(height, width),
                                score_mean=float(values.mean(dtype=np.float64)), score_max=float(values.max()),
                                score_p95=float(np.percentile(values, 95)),
                                region_aupro_005=region_pro[0.05], region_aupro_030=region_pro[0.30]))
    return result, None, region_rows


def group_analysis(samples, np):
    groups = []
    for group, label in GROUPS:
        subset = [row for row in samples if row["size_group"] == group]
        result = dict(group=group, range=label, sample_count=len(subset))
        for name in GROUP_METRICS:
            values = np.array([row[name] for row in subset], dtype=np.float64)
            for stat, function in (("mean", np.mean), ("median", np.median), ("std", np.std), ("min", np.min), ("max", np.max)):
                result[f"{name}_{stat}"] = float(function(values)) if len(values) else None
        groups.append(result)
    return groups


def hypotheses(samples, normals, regions, np, report):
    try:
        from scipy.stats import spearmanr, mannwhitneyu, rankdata
    except ImportError:
        spearmanr = mannwhitneyu = rankdata = None
        report["warnings"].append("scipy NOT_AVAILABLE; rank tests skipped")

    def correlation(x, y):
        if spearmanr is None:
            return dict(status="NOT_AVAILABLE", rho=None, p=None)
        if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
            return dict(status="INCONCLUSIVE", rho=None, p=None)
        rho, p = spearmanr(x, y)
        return dict(status="OK", rho=float(rho), p=float(p), n=len(x))

    ratio = [row["defect_ratio"] for row in samples]
    for metric in ("aupro_005", "pixel_auroc"):
        report["correlations"][f"defect_ratio_vs_{metric}"] = correlation(ratio, [row[metric] for row in samples])
    tiny = [row["aupro_005"] for row in samples if row["size_group"] == "A"]
    large = [row["aupro_005"] for row in samples if row["size_group"] in ("D", "E")]
    h1 = dict(status="INCONCLUSIVE", tiny_n=len(tiny), large_n=len(large), comparison="A vs pooled D/E")
    if tiny and large:
        h1.update(tiny_mean=float(np.mean(tiny)), large_mean=float(np.mean(large)),
                  mean_difference_large_minus_tiny=float(np.mean(large)-np.mean(tiny)),
                  median_difference_large_minus_tiny=float(np.median(large)-np.median(tiny)))
    if len(tiny) >= 3 and len(large) >= 3:
        p = float(mannwhitneyu(tiny, large, alternative="two-sided").pvalue) if mannwhitneyu else None
        h1["exploratory_p"] = p
        if h1["mean_difference_large_minus_tiny"] > 0 and h1["median_difference_large_minus_tiny"] > 0:
            h1["status"] = "SUPPORTED" if p is not None and p < 0.05 else "PARTIALLY_SUPPORTED"
        elif h1["mean_difference_large_minus_tiny"] <= 0 and h1["median_difference_large_minus_tiny"] <= 0:
            h1["status"] = "NOT_SUPPORTED"
    h2 = dict(report["correlations"]["defect_ratio_vs_aupro_005"])
    rho, p = h2["rho"], h2["p"]
    h2["status"] = "INCONCLUSIVE" if rho is None else ("NOT_SUPPORTED" if rho <= 0 else
                        "SUPPORTED" if p < 0.05 else "PARTIALLY_SUPPORTED")
    a = np.array([row["global_max"] for row in samples])
    n = np.array([row["max_score"] for row in normals])
    overlap = float(((n[:, None] > a).astype(float) + 0.5*(n[:, None] == a)).mean())
    above = int((n > np.percentile(a, 95)).sum())
    h3 = dict(status="PARTIALLY_SUPPORTED" if above > 0 and overlap >= 0.25 else "INCONCLUSIVE",
              normal_over_anomaly_pairwise_rank_overlap=overlap, normals_above_anomaly_score_q95=above,
              normal_max_score_median=float(np.median(n)), anomaly_max_score_median=float(np.median(a)),
              note="Score overlap only; no selected operating threshold, FP rate or severity conclusion.")
    distance = [row["border_distance_normalized"] for row in regions]
    performance = [row["region_aupro_005"] for row in regions]
    h4 = dict(status="INCONCLUSIVE", association=correlation(distance, performance),
              region_count=len(regions), partial_rank_correlation_controlling_area=None)
    if rankdata is not None and h4["association"]["rho"] is not None:
        design = np.column_stack([np.ones(len(regions)), rankdata([row["region_pixel_count"] for row in regions])])
        x, y = rankdata(distance), rankdata(performance)
        rx = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
        ry = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
        if np.std(rx) > 1e-10 and np.std(ry) > 1e-10:
            partial = float(np.corrcoef(rx, ry)[0, 1])
            h4["partial_rank_correlation_controlling_area"] = partial
            if partial > 0 and h4["association"]["rho"] > 0:
                h4["status"] = "PARTIALLY_SUPPORTED"
            elif partial <= 0 and h4["association"]["rho"] <= 0:
                h4["status"] = "NOT_SUPPORTED"
    report["hypotheses"] = dict(H1=h1, H2=h2, H3=h3, H4=h4)


def figures(samples, normals, groups, np, plt):
    for metric, name, ylabel in (("aupro_005", "defect_ratio_vs_aupro005.png", "Per-image AU-PRO@0.05 (%)"),
                                  ("pixel_auroc", "defect_ratio_vs_pixel_auroc.png", "Per-image Pixel AUROC (%)")):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter([row["defect_ratio"]*100 for row in samples], [row[metric]*100 for row in samples], alpha=0.7)
        ax.set(xscale="log", xlabel="Defect area (% of image)", ylabel=ylabel)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(OUT / "figures" / name, dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, row in enumerate(groups):
        if row["sample_count"]:
            ax.bar(i, row["aupro_005_mean"]*100, yerr=row["aupro_005_std"]*100, capsize=5)
        ax.text(i, 1.02, f"n={row['sample_count']}", ha="center", transform=ax.get_xaxis_transform())
    ax.set_xticks(range(5), [row["range"] for row in groups])
    ax.set(xlabel="Fixed defect area group", ylabel="Mean per-image AU-PRO@0.05 (%)")
    fig.tight_layout()
    fig.savefig(OUT / "figures/size_group_aupro005.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist([row["max_score"] for row in normals], bins=8, edgecolor="white")
    ax.set(xlabel="Normal-image maximum anomaly score", ylabel="Image count",
           title="Normal responses (no operating threshold applied)")
    fig.tight_layout()
    fig.savefig(OUT / "figures/normal_max_score_distribution.png", dpi=180)
    plt.close(fig)


def galleries(samples, normals, inputs, np, torch, Image, plt, report):
    mapping = {row["sample_id"]: row for row in inputs}
    albums = {"worst_10": sorted(samples, key=lambda row: (row["aupro_005"], row["sample_id"]))[:10],
              "best_10": sorted(samples, key=lambda row: (-row["aupro_005"], row["sample_id"]))[:10],
              "smallest_10": sorted(samples, key=lambda row: (row["defect_ratio"], row["sample_id"]))[:10],
              "highest_normal_response_10": normals[:10]}
    report["gallery_selection"] = {name: [row["sample_id"] for row in selected] for name, selected in albums.items()}
    report["gallery_png_count"] = 0
    for name, selected in albums.items():
        folder = OUT / "galleries" / name
        folder.mkdir(exist_ok=True)
        for row in selected:
            original = mapping[row["sample_id"]]
            native = exact.get_score(original, torch, np)
            mask = exact.get_mask(original, Image, np)
            with Image.open(row["image_path"]) as image:
                size = (1200, max(1, round(image.height*1200/image.width)))
                rgb = np.asarray(image.convert("RGB").resize(size, Image.Resampling.BILINEAR))
            display = np.asarray(Image.fromarray(native).resize(size, Image.Resampling.BILINEAR))
            display = (display-display.min()) / max(float(display.max()-display.min()), 1e-12)
            gt = np.asarray(Image.fromarray(mask*255).resize(size, Image.Resampling.NEAREST))
            fig, axes = plt.subplots(2, 2, figsize=(16, 6), constrained_layout=True)
            try:
                for ax, title in zip(axes.flat, ("Original", "GT mask", "Native map (display-normalized)", "Overlay")):
                    ax.set_title(title)
                    ax.axis("off")
                axes[0, 0].imshow(rgb)
                axes[0, 1].imshow(gt, cmap="gray", vmin=0, vmax=255)
                axes[1, 0].imshow(display, cmap="jet", vmin=0, vmax=1)
                axes[1, 1].imshow(rgb)
                axes[1, 1].imshow(display, cmap="jet", vmin=0, vmax=1, alpha=0.45)
                if "defect_ratio" in row:
                    caption = (f"{row['sample_id']} | defect={100*row['defect_ratio']:.5f}% | "
                               f"Pixel AUROC={100*row['pixel_auroc']:.2f}% | AU-PRO@0.05={100*row['aupro_005']:.2f}%")
                else:
                    caption = f"{row['sample_id']} | normal | max score={row['max_score']:.6f}"
                fig.suptitle(caption)
                fig.savefig(folder / f"{row['sample_id']}.png", dpi=150)
            finally:
                plt.close(fig)
            report["gallery_png_count"] += 1


def summary(report):
    lines = ["="*40, "PHASE 07D - B0 FAILURE ANALYSIS", "="*40, "B0: EfficientAD-S / 256x256",
             f"Anomaly samples: {report.get('anomaly_samples', 'N/A')}",
             f"Connected regions: {report.get('connected_regions', 'N/A')}", "\nDEFECT SIZE GROUPS"]
    for group in report.get("size_groups", []):
        value = group["aupro_005_mean"]
        lines.append(f"{group['range']}: n={group['sample_count']}, mean AU-PRO@0.05=" +
                     ("N/A" if value is None else f"{100*value:.2f}%"))
    lines.append("\nSpearman: Defect ratio vs AU-PRO@0.05")
    lines.append(str(report["correlations"].get("defect_ratio_vs_aupro_005", "NOT_RUN")))
    lines.append("\nHighest-response normal samples (no threshold-defined FP classification):")
    for row in report.get("highest_normal_responses", []):
        lines.append(f"{row['sample_id']}: {row['max_score']:.6f}")
    for key, evidence in report["hypotheses"].items():
        lines.append(f"\n{key}: {evidence['status']}\nEvidence: {json.dumps(evidence, ensure_ascii=False)}")
    lines.extend(["\nSTATUS:", report["status"]])
    if report["error_message"]:
        lines.append(report["error_message"])
    return "\n".join(lines + ["="*40])


def save_reports(report):
    failures = []
    for name in ("failure_analysis.json", "failure_analysis.txt"):
        try:
            if name.endswith("json"):
                exact.write_json(OUT / name, report)
            else:
                (OUT / name).write_text(summary(report)+"\n\n"+json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            traceback.print_exc()
            failures.append(str(exc))
    if failures:
        raise OSError("; ".join(failures))


def main():
    report = initial_report()
    report["timestamp"] = datetime.now().astimezone().isoformat()
    started = time.perf_counter()
    stage = "INPUT"
    code = 1
    try:
        os.chdir(ROOT)
        prepare_outputs()
        exact.write_json(OUT / "B0_BASELINE.json", BASELINE)
        import numpy as np
        import torch
        import cv2
        from PIL import Image
        from sklearn.metrics import roc_auc_score
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        inputs = mapping_check(np, Image, report)
        samples, normals, regions = [], [], []
        stage = "ANALYSIS"
        for index, row in enumerate(inputs):
            sample, normal, region_rows = analyze_sample(row, np, torch, cv2, Image, roc_auc_score)
            if sample is not None:
                samples.append(sample)
            if normal is not None:
                normals.append(normal)
            regions.extend(region_rows)
            print(f"Analyzed saved prediction {index+1}/114", flush=True)
        normals.sort(key=lambda row: (-row["max_score"], row["sample_id"]))
        for rank, row in enumerate(normals, 1):
            row["rank"] = rank
        cutpoints = np.quantile([row["region_pixel_count"] for row in regions], [0.2, 0.4, 0.6, 0.8])
        names = ["very_tiny", "tiny", "small", "medium", "large"]
        for row in regions:
            row["area_quantile_group"] = names[int(np.searchsorted(cutpoints, row["region_pixel_count"], side="left"))]
        report.update(connected_regions=len(regions), region_area_quantiles=dict(zip(["Q20", "Q40", "Q60", "Q80"], cutpoints.tolist())),
                      highest_normal_responses=normals[:10], size_groups=group_analysis(samples, np))
        hypotheses(samples, normals, regions, np, report)
        stage = "OUTPUT"
        write_csv("sample_analysis.csv", samples, SAMPLE_FIELDS)
        write_csv("normal_fp_analysis.csv", normals, NORMAL_FIELDS)
        write_csv("region_analysis.csv", regions, REGION_FIELDS)
        write_csv("size_group_analysis.csv", report["size_groups"], GROUP_FIELDS)
        report["status"] = "ANALYSIS_COMPLETE_FIGURES_PENDING"
        save_reports(report)
        stage = "FIGURES"
        figures(samples, normals, report["size_groups"], np, plt)
        galleries(samples, normals, inputs, np, torch, Image, plt, report)
        stage = "INTEGRITY"
        for item in report["input_integrity"]["input_raw_map_hashes"]:
            if exact.digest(Path(item["path"])) != item["sha256"]:
                raise ValueError(f"Raw map changed: {item['path']}")
        if exact.digest(exact.SOURCE / "predictions.csv") != report["input_integrity"]["input_csv_sha256"]:
            raise ValueError("Input CSV changed")
        report["status"] = "PASS"
        code = 0
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status="FAIL", failure_stage=stage, error_message=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        report["elapsed_seconds"] = time.perf_counter()-started
        try:
            prepare_outputs()
            save_reports(report)
        except Exception as exc:
            traceback.print_exc()
            report.update(status="FAIL", error_message=f"Report writing failed: {exc}")
            code = 1
        print(summary(report), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
