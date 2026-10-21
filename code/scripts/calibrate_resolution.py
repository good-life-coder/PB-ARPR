"""Phase 08A: sequential 384/512 resolution calibration, 500 steps each.

Run manually in metal_ad_ead. No evaluation or checkpoint saving.
Only the official EfficientAD resize changes relative to 256 preprocessing.
"""

import gc
import inspect
import json
import math
import os
import platform
import sys
import time
import traceback
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/phase08a_resolution_calibration"
RESOLUTIONS = (384, 512)
MIB = 1024 ** 2
B0_HOURS = 4.2807
B0_ALLOCATED_MIB = 1185.2217


def result_template(size):
    return dict(resolution=[size, size], target_steps=500, completed_steps=0,
                seed=42, batch_size=1, model_size="small", max_epochs=1000,
                precision="32-true", warmup_steps=50, steady_target_steps=450,
                total_training_seconds=None, steady_training_seconds=None,
                total_seconds_per_step=None, steady_seconds_per_step=None,
                projected_70000_hours_total=None, projected_70000_hours_steady=None,
                peak_allocated_mib=None, peak_reserved_mib=None, gpu_total_memory_mib=None,
                allocated_fraction_gpu=None, reserved_fraction_gpu=None,
                allocated_percent_gpu=None, reserved_percent_gpu=None,
                time_ratio_vs_256=None, time_ratio_vs_256_total=None, memory_ratio_vs_256=None,
                nan_detected=False, inf_detected=False, oom_detected=False, dataloader_error=False,
                loss_checks=0, initial_losses=[], final_losses=[], feasible_under_8gb=None,
                status="NOT_RUN", failure_stage=None, error_message=None, traceback=None, warnings=[])


def initial_report():
    return dict(timestamp=datetime.now().astimezone().isoformat(), status="NOT_RUN",
                b0_training_hours=B0_HOURS, b0_peak_allocated_mib=B0_ALLOCATED_MIB,
                dataset="MVTec AD 2", category="sheet_metal", test_type="public", gpu_index=0,
                runs=[result_template(size) for size in RESOLUTIONS], error_message=None,
                notes=["Projection is linear, not a guarantee of final training duration.",
                       "Fit time includes official teacher/ImageNette setup and teacher statistics.",
                       "Steady timer starts after optimizer step 50 and ends after step 500, with CUDA synchronization.",
                       "Loss finiteness checks synchronize tensors and add a small measurement overhead.",
                       "B0 time is a completed formal run; these are calibration projections, not matched formal timings.",
                       "GPU fractions are 0..1; percentage fields are 0..100.",
                       "Feasible means this 500-step FP32 run passed within GPU/8 GiB capacity; it does not guarantee a full run.",
                       "No final-experiment recommendation is made."])


def summary(report):
    lines = ["="*40, "PHASE 08A - RESOLUTION CALIBRATION", "="*40,
             "B0 256:", "Training time = 4.2807 h", "Peak VRAM = 1185.22 MiB"]
    for r in report["runs"]:
        lines.extend(["-"*40, f"{r['resolution'][0]}x{r['resolution'][1]}", "-"*40])
        for label, key in (("Completed steps", "completed_steps"), ("Steady seconds/step", "steady_seconds_per_step"),
                           ("Projected 70000 steps (steady hours)", "projected_70000_hours_steady"),
                           ("Projected 70000 steps (total-average hours)", "projected_70000_hours_total"),
                           ("Peak allocated (MiB)", "peak_allocated_mib"), ("Peak reserved (MiB)", "peak_reserved_mib"),
                           ("Time ratio vs 256", "time_ratio_vs_256"), ("VRAM ratio vs 256", "memory_ratio_vs_256"),
                           ("Status", "status")):
            value = r[key]
            text = "N/A" if value is None else f"{value:.6f}" if isinstance(value, float) else str(value)
            lines.append(f"{label}: {text}")
        if r["error_message"]:
            lines.append(f"{r['failure_stage']}: {r['error_message']}")
    lines.extend(["-"*40, "RECOMMENDATION DATA ONLY:"])
    for r in report["runs"]:
        lines.append(f"{r['resolution'][0]} feasible under 8GB: {r['feasible_under_8gb']}")
    lines.extend(["No final experiment automatically recommended.", "STATUS:", report["status"]])
    if report["error_message"]:
        lines.append(report["error_message"])
    return "\n".join(lines + ["="*40])


def save_report(report):
    OUT.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    failures = []
    for name, content in (("resolution_calibration.json", text),
                          ("resolution_calibration.txt", summary(report)+"\n\n"+text)):
        try:
            (OUT / name).write_text(content+"\n", encoding="utf-8")
        except OSError as exc:
            traceback.print_exc()
            failures.append(str(exc))
    if failures:
        raise OSError("Report writing failed: " + "; ".join(failures))


def official_preprocessor(EfficientAd, size):
    """Inspect official API and verify the transform attributes except resize size."""
    signature = inspect.signature(EfficientAd.configure_pre_processor)
    source = inspect.getsource(EfficientAd.configure_pre_processor)
    if "image_size" not in signature.parameters:
        raise RuntimeError("Official image_size API unavailable")
    default = EfficientAd.configure_pre_processor()
    selected = EfficientAd.configure_pre_processor(image_size=(size, size))
    before, after = default.transform.transforms, selected.transform.transforms
    if len(before) != 1 or len(after) != 1 or type(before[0]) is not type(after[0]) or type(before[0]).__name__ != "Resize":
        raise RuntimeError("Expected identical single-Resize pipelines; refusing preprocessing changes")
    if tuple(before[0].size) != (256, 256) or tuple(after[0].size) != (size, size):
        raise RuntimeError("Unexpected preprocessing dimensions")
    before_config = {k: repr(v) for k, v in vars(before[0]).items() if k != "size"}
    after_config = {k: repr(v) for k, v in vars(after[0]).items() if k != "size"}
    if before_config != after_config:
        raise RuntimeError("Non-size Resize configuration differs from default")
    return selected, dict(api="EfficientAd.configure_pre_processor(image_size=(resolution, resolution))",
                          signature=str(signature), source=source, default_transform=repr(default.transform),
                          selected_transform=repr(selected.transform), non_size_attributes=after_config,
                          only_resize_changed=True)


def make_monitor(Callback, torch, result):
    class Monitor(Callback):
        def __init__(self):
            self.steady_start = None

        def on_before_backward(self, trainer, pl_module, loss):
            value = float(loss.detach().item())
            result["loss_checks"] += 1
            result["nan_detected"] |= math.isnan(value)
            result["inf_detected"] |= math.isinf(value)
            entry = dict(step=int(trainer.global_step)+1, loss=value if math.isfinite(value) else str(value))
            if len(result["initial_losses"]) < 10:
                result["initial_losses"].append(entry)
            result["final_losses"].append(entry)
            del result["final_losses"][:-10]
            if not math.isfinite(value):
                raise FloatingPointError(f"Nonfinite loss before step {entry['step']}: {value}")

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            step = int(trainer.global_step)
            result["completed_steps"] = step
            if step == 50:
                torch.cuda.synchronize()
                self.steady_start = time.perf_counter()
            if step == 500:
                torch.cuda.synchronize()
                if self.steady_start is None:
                    raise RuntimeError("Missing warm-up boundary")
                result["steady_training_seconds"] = time.perf_counter()-self.steady_start
            if step % 100 == 0:
                print(f"{result['resolution'][0]}: completed {step}/500 optimizer steps", flush=True)
    return Monitor()


def run_one(size, result, torch, EfficientAd, MVTecAD2, Engine, Callback, seed_everything):
    model = engine = datamodule = monitor = hook = None
    fit_start = None
    memory_ready = False
    stage = "DATA"
    try:
        seed_everything(42, workers=True)
        datamodule = MVTecAD2(root="./datasets/MVTec_AD_2", category="sheet_metal", train_batch_size=1,
                             eval_batch_size=1, num_workers=0, test_type="public")
        if not (ROOT / "datasets/MVTec_AD_2/sheet_metal").is_dir():
            raise FileNotFoundError("Existing dataset required; no MVTec download allowed")
        datamodule.setup()
        for key, data, expected in (("train_count", datamodule.train_data, 137),
                                    ("validation_count", datamodule.val_data, 19),
                                    ("test_public_count", datamodule.test_public_data, 114)):
            result[key] = len(data)
            if len(data) != expected:
                raise ValueError(f"{key}: expected {expected}, found {len(data)}")
        stage = "MODEL_INIT"
        preprocessor, result["preprocessor"] = official_preprocessor(EfficientAd, size)
        model = EfficientAd(model_size="small", imagenet_dir="./datasets/imagenette", pre_processor=preprocessor,
                            evaluator=False, visualizer=False, post_processor=False)
        result.update(learning_rate=model.lr, weight_decay=model.weight_decay,
                      parameter_count=sum(p.numel() for p in model.parameters()))

        def check_forward_shape(module, args, kwargs):
            batch = kwargs.get("batch", args[0] if args else None)
            if batch is None or tuple(batch.shape[-2:]) != (size, size):
                raise RuntimeError("Actual model input differs from requested resolution")
            result["actual_model_input_size"] = list(batch.shape[-2:])
        hook = model.model.register_forward_pre_hook(check_forward_shape, with_kwargs=True)
        monitor = make_monitor(Callback, torch, result)
        engine = Engine(max_steps=500, max_epochs=1000, accelerator="gpu", devices=1,
                        precision="32-true", limit_train_batches=1.0, limit_val_batches=0,
                        num_sanity_val_steps=0, enable_checkpointing=False, logger=False, barebones=True,
                        default_root_dir=str(OUT / str(size)), callbacks=[monitor])
        stage = "CUDA"
        total_memory = torch.cuda.get_device_properties(0).total_memory / MIB
        result["gpu_total_memory_mib"] = total_memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        memory_ready = True
        result["pre_allocated_mib"] = torch.cuda.memory_allocated() / MIB
        result["pre_reserved_mib"] = torch.cuda.memory_reserved() / MIB
        torch.cuda.synchronize()
        stage = "TRAINING"
        fit_start = time.perf_counter()
        engine.fit(model=model, datamodule=datamodule)
        torch.cuda.synchronize()
        result["total_training_seconds"] = time.perf_counter()-fit_start
        result["completed_steps"] = int(engine.trainer.global_step)
        if result["completed_steps"] != 500 or result["loss_checks"] != 500:
            raise RuntimeError("Expected exactly 500 optimizer steps and finite-loss checks")
        if result["steady_training_seconds"] is None or result["steady_training_seconds"] <= 0:
            raise RuntimeError("Missing steady-state timing")
        result["steady_seconds_per_step"] = result["steady_training_seconds"] / 450
        result["projected_70000_hours_steady"] = result["steady_seconds_per_step"] * 70000 / 3600
        result["time_ratio_vs_256"] = result["projected_70000_hours_steady"] / B0_HOURS
        result["status"] = "PASS"
    except (Exception, KeyboardInterrupt) as exc:
        detail = traceback.format_exc()
        result["oom_detected"] = isinstance(exc, torch.cuda.OutOfMemoryError) or "cuda out of memory" in str(exc).lower()
        result["dataloader_error"] = "dataloader" in detail.lower() or "data_loader" in detail.lower()
        result.update(status="OOM" if result["oom_detected"] else "FAIL", failure_stage=stage,
                      error_message=f"{type(exc).__name__}: {exc}", traceback=detail)
        traceback.print_exc()
    finally:
        if fit_start is not None and result["total_training_seconds"] is None:
            result["total_training_seconds"] = time.perf_counter()-fit_start
        if engine is not None:
            try:
                result["completed_steps"] = int(engine.trainer.global_step)
            except Exception as exc:
                result["warnings"].append(f"Trainer step count unavailable: {exc}")
        if result["completed_steps"] == 500 and result["total_training_seconds"] is not None:
            result["total_seconds_per_step"] = result["total_training_seconds"] / 500
            result["projected_70000_hours_total"] = result["total_seconds_per_step"] * 70000 / 3600
            result["time_ratio_vs_256_total"] = result["projected_70000_hours_total"] / B0_HOURS
        if memory_ready:
            try:
                result["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / MIB
                result["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / MIB
                result["allocated_fraction_gpu"] = result["peak_allocated_mib"] / result["gpu_total_memory_mib"]
                result["reserved_fraction_gpu"] = result["peak_reserved_mib"] / result["gpu_total_memory_mib"]
                result["allocated_percent_gpu"] = result["allocated_fraction_gpu"] * 100
                result["reserved_percent_gpu"] = result["reserved_fraction_gpu"] * 100
                result["memory_ratio_vs_256"] = result["peak_allocated_mib"] / B0_ALLOCATED_MIB
            except Exception as exc:
                traceback.print_exc()
                result["warnings"].append(f"VRAM query failed: {exc}")
                if result["status"] == "PASS":
                    result.update(status="FAIL", failure_stage="CUDA", error_message=str(exc))
        if hook is not None:
            hook.remove()
        # Drop Trainer/optimizer/callback references as well as the model before next run.
        del hook, engine, monitor, datamodule, model
        gc.collect()
        try:
            torch.cuda.empty_cache()
            result["post_cleanup_allocated_mib"] = torch.cuda.memory_allocated() / MIB
            result["post_cleanup_reserved_mib"] = torch.cuda.memory_reserved() / MIB
        except Exception as exc:
            traceback.print_exc()
            result["warnings"].append(f"GPU cleanup failed: {exc}")
            result.update(status="FAIL", failure_stage="CLEANUP", error_message=str(exc))
        if result["status"] == "PASS":
            result["feasible_under_8gb"] = (result["peak_reserved_mib"] <= min(8192, result["gpu_total_memory_mib"]))
        elif result["status"] == "OOM":
            result["feasible_under_8gb"] = False


def main():
    report = initial_report()
    started = time.perf_counter()
    code = 1
    try:
        os.chdir(ROOT)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        import torch
        from lightning.pytorch import Callback, seed_everything
        from anomalib.models import EfficientAd
        from anomalib.data import MVTecAD2
        from anomalib.engine import Engine
        if version("anomalib") != "2.6.1":
            raise RuntimeError("Expected anomalib 2.6.1; verify preprocessing API before changing version")
        report.update(python_version=platform.python_version(), torch_version=str(torch.__version__),
                      cuda_runtime=torch.version.cuda, anomalib_version=version("anomalib"),
                      cudnn_deterministic=torch.backends.cudnn.deterministic,
                      cudnn_benchmark=torch.backends.cudnn.benchmark)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU 0 required")
        torch.cuda.set_device(0)
        report["gpu_name"] = torch.cuda.get_device_name(0)
        for size, result in zip(RESOLUTIONS, report["runs"]):
            run_one(size, result, torch, EfficientAd, MVTecAD2, Engine, Callback, seed_everything)
            save_report(report)
            if result["status"] != "PASS":
                for pending in report["runs"]:
                    if pending["status"] == "NOT_RUN":
                        pending.update(status="SKIPPED", error_message=f"Stopped after {size} failure")
                break
        report["status"] = "PASS" if all(r["status"] == "PASS" for r in report["runs"]) else "FAIL"
        code = 0 if report["status"] == "PASS" else 1
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status="FAIL", error_message=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        report["elapsed_seconds"] = time.perf_counter()-started
        try:
            save_report(report)
        except Exception as exc:
            traceback.print_exc()
            report.update(status="FAIL", error_message=f"Report writing failed: {exc}")
            code = 1
        print(summary(report), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
