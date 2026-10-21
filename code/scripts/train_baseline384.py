"""Phase 08B 384-resolution baseline. Run manually; never evaluates public test data.

python scripts/train_baseline384.py
python scripts/train_baseline384.py --resume
The latter restores only this experiment's official checkpoints/last.ckpt.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from lightning.pytorch.callbacks import Callback

try:
    from .calibrate_resolution import official_preprocessor
except ImportError:
    from calibrate_resolution import official_preprocessor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "phase08b_baseline384_seed42"
FINAL = OUT / "checkpoints" / "final.ckpt"
LAST = OUT / "checkpoints" / "last.ckpt"
LOSSES = ("train_loss", "train_st", "train_ae", "train_stae")
MIB = 1024 ** 2
B0_DIR = ROOT / "experiments" / "phase07b_baseline256_seed42"
B0_HOURS = 4.2807
B0_VRAM_MIB = 1185.2217


def now():
    return datetime.now().astimezone().isoformat()


def protocol():
    return {
        "name": "B1_resolution384", "baseline_reference": "B0",
        "training_steps": 70000,
        "difference_from_B0": "input resolution only: 256x256 -> 384x384",
        "dataset": "MVTec AD 2", "dataset_root": str(ROOT / "datasets" / "MVTec_AD_2"),
        "category": "sheet_metal", "train_count": 137, "validation_count": 19,
        "test_public_count": 114, "test_type": "public", "model": "EfficientAD-S",
        "model_size": "small", "input_size": [384, 384], "batch_size": 1,
        "num_workers": 0, "seed": 42, "max_steps": 70000, "max_epochs": 1000,
        "accelerator": "gpu", "devices": 1, "gpu_index": 0, "precision": "32-true",
        "optimizer": "Adam", "learning_rate": 0.0001, "weight_decay": 0.00001,
        "scheduler": "StepLR", "scheduler_step_size": 66500, "scheduler_gamma": 0.1,
        "validation_during_fit": False, "final_full_validation": True,
        "evaluator": False, "visualizer": False, "post_processor": False,
        "pre_processor": "Official EfficientAD configure_pre_processor(image_size=(384, 384))",
        "early_stopping": False, "checkpoint_selection": "final step plus final validation",
    }


def initial_report():
    return {
        "status": "NOT_RUN", "timestamp_start": None, "timestamp_end": None,
        "target_steps": 70000, "completed_steps": 0, "global_step": 0, "final_epoch": None,
        "training_elapsed_seconds": None, "training_elapsed_minutes": None,
        "training_elapsed_hours": None, "session_training_elapsed_seconds": None,
        "total_session_elapsed_seconds": None, "seconds_per_step": None,
        "validation_elapsed_seconds": None, "final_validation_completed": False,
        "final_validation_batches": 0, "final_quantiles": None,
        "peak_gpu_allocated_mib": None, "peak_gpu_reserved_mib": None,
        "gpu_name": None, "gpu_total_memory_mib": None,
        "checkpoint_path": None, "checkpoint_size_mb": None, "checkpoint_sha256": None,
        "training_time_ratio_vs_b0": None, "training_vram_ratio_vs_b0": None,
        "loss_nan_detected": False, "loss_inf_detected": False,
        "loss_fields_present": {name: False for name in LOSSES},
        "loss_health_checked": False, "resume": False, "resume_checkpoint": None,
        "resume_start_step": 0, "log_paths": [], "failure_stage": None,
        "error_message": None, "traceback": None, "warnings": [],
        "timing_note": "Training time includes engine.fit setup/statistics. On resume it sums "
                       "saved fit time and current fit time; discarded work after last.ckpt is excluded. "
                       "VRAM peaks describe this process. No public test metrics are computed.",
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def summary(r):
    def fmt(key):
        return "N/A" if r[key] is None else f"{r[key]:.4f}"
    lines = ["=" * 40, "PHASE 08B - FORMAL 384 TRAINING", "=" * 40,
             "Dataset: MVTec AD 2 / sheet_metal", "Model: EfficientAD-S", "Input: 384x384",
             "Seed: 42", "Batch size: 1", "", "Target steps: 70000",
             f"Completed steps: {r['completed_steps']}", "", "Training time:",
             f"{fmt('training_elapsed_seconds')} s", f"{fmt('training_elapsed_minutes')} min",
             f"{fmt('training_elapsed_hours')} h", "", "Seconds / step:", fmt("seconds_per_step"),
             "", "Peak allocated VRAM:", f"{fmt('peak_gpu_allocated_mib')} MiB",
             "", "Peak reserved VRAM:", f"{fmt('peak_gpu_reserved_mib')} MiB",
             "", "Training time ratio vs B0:", fmt("training_time_ratio_vs_b0"),
             "VRAM ratio vs B0:", fmt("training_vram_ratio_vs_b0"),
             "", "Final checkpoint:", str(r["checkpoint_path"]), "", "Checkpoint SHA256:",
             str(r["checkpoint_sha256"]), "", f"NaN detected: {r['loss_nan_detected']}",
             f"Inf detected: {r['loss_inf_detected']}", "", "STATUS:", r["status"]]
    if r["failure_stage"]:
        lines.extend([f"Failure stage: {r['failure_stage']}", str(r["error_message"])])
    return "\n".join(lines + ["=" * 40])


def write_reports(r):
    OUT.mkdir(parents=True, exist_ok=True)
    errors = []
    for name in ("training_report.json", "training_report.txt"):
        try:
            if name.endswith("json"):
                write_json(OUT / name, r)
            else:
                (OUT / name).write_text(summary(r) + "\n\n" + json.dumps(r, indent=2, ensure_ascii=False,
                                                                          allow_nan=False) + "\n", encoding="utf-8")
        except Exception as exc:
            traceback.print_exc()
            errors.append(str(exc))
    if errors:
        raise OSError("Could not save reports: " + "; ".join(errors))


def environment_snapshot(extra):
    lines = [f"Snapshot: {now()}", f"Python executable: {sys.executable}"]
    lines.extend(f"{key}: {value}" for key, value in extra.items())
    for command in ([sys.executable, "--version"], [sys.executable, "-m", "pip", "freeze"],
                    ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv"],
                    ["nvidia-smi"]):
        lines.append("\n$ " + " ".join(command))
        try:
            result = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=90,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False)
            lines.extend([result.stdout, result.stderr, f"exit_code: {result.returncode}"])
        except (OSError, subprocess.TimeoutExpired) as exc:
            lines.append(f"Snapshot command unavailable: {exc}")
    (OUT / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_b0_consistency(config, versions, gpu_name):
    """Fail closed on experimental-protocol or environment drift; read B0 only."""
    frozen = json.loads((B0_DIR / "config.json").read_text(encoding="utf-8"))
    changed = {}
    for key, value in protocol().items():
        if key in frozen and value != frozen[key]:
            changed[key] = {"b0": frozen[key], "b1": value}
    if set(changed) != {"input_size", "pre_processor"}:
        raise ValueError(f"Unexpected experimental protocol differences: {changed}")
    if frozen["input_size"] != [256, 256] or config["input_size"] != [384, 384]:
        raise ValueError("Resolution ablation must be 256 -> 384")
    for key, value in versions.items():
        if key not in frozen or value != frozen[key]:
            raise ValueError(f"B0 environment mismatch: {key}: B0={frozen.get(key)}, current={value}")
    frozen_env = (B0_DIR / "environment.txt").read_text(encoding="utf-8")
    saved_gpu = re.search(r"^gpu_name: (.+)$", frozen_env, re.MULTILINE)
    if saved_gpu is None or saved_gpu.group(1).strip() != gpu_name:
        raise ValueError("GPU differs from B0 or B0 GPU identity is missing")
    saved_driver = re.search(r"^0,\s*[^,]+,\s*([^,\r\n]+)", frozen_env, re.MULTILINE)
    result = subprocess.run(["nvidia-smi", "--id=0", "--query-gpu=driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True, timeout=20, check=True,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    driver = result.stdout.strip()
    if saved_driver is None or saved_driver.group(1).strip() != driver:
        raise ValueError(f"NVIDIA driver differs from B0: current={driver}")
    return dict(status="PASS", protocol_differences=changed, framework_versions_match=True,
                gpu_match=True, nvidia_driver_version=driver, nvidia_driver_match=True,
                b0_config_sha256=hashlib.sha256((B0_DIR / "config.json").read_bytes()).hexdigest())


def check_logs(paths, report):
    """Ignore empty CSV cells and accept Lightning's _step/_epoch column suffixes."""
    present = {name: False for name in LOSSES}
    for path in paths:
        with Path(path).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                for name in LOSSES:
                    for column in (name, name + "_step", name + "_epoch"):
                        text = row.get(column)
                        if text is None or not text.strip():
                            continue
                        value = float(text)
                        present[name] = True
                        report["loss_nan_detected"] |= math.isnan(value)
                        report["loss_inf_detected"] |= math.isinf(value)
    report["loss_fields_present"] = present
    report["loss_health_checked"] = True
    if not all(present.values()) or report["loss_nan_detected"] or report["loss_inf_detected"]:
        raise ValueError(f"Loss health check failed: fields={present}, "
                         f"NaN={report['loss_nan_detected']}, Inf={report['loss_inf_detected']}")


class BaselineMonitor(Callback):
    """Module-level pickle-safe monitor; stores only Python dictionaries/scalars."""

    def __init__(self, report, config, timing):
        super().__init__()
        self.report = report
        self.config = config
        self.timing = timing
        self.last_loss = None

    @property
    def state_key(self) -> str:
        return "BaselineMonitor"

    def state_dict(self):
        return {"nan_detected": bool(self.report["loss_nan_detected"]),
                "inf_detected": bool(self.report["loss_inf_detected"]),
                "last_loss": self.last_loss}

    def load_state_dict(self, state_dict):
        self.report["loss_nan_detected"] = bool(state_dict.get("nan_detected", False))
        self.report["loss_inf_detected"] = bool(state_dict.get("inf_detected", False))
        value = state_dict.get("last_loss")
        self.last_loss = None if value is None else float(value)

    def on_before_backward(self, trainer, pl_module, loss):
        value = float(loss.detach().item())
        self.last_loss = value
        self.report["loss_nan_detected"] |= math.isnan(value)
        self.report["loss_inf_detected"] |= math.isinf(value)
        if not math.isfinite(value):
            raise FloatingPointError(f"Nonfinite train_loss before step {trainer.global_step + 1}: {value}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.report["completed_steps"] = int(trainer.global_step)
        if trainer.global_step % 1000 == 0:
            print(f"Baseline training: {trainer.global_step}/70000 steps", flush=True)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.report["final_validation_batches"] += 1

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        timing = self.timing
        elapsed = timing["prior"] + (time.perf_counter() - timing["start"] if timing["start"] else 0)
        checkpoint["phase08b"] = {
            "protocol": protocol(), "config": self.config,
            "training_elapsed_seconds": timing["finished"] if timing["finished"] is not None else elapsed,
            "timestamp_start": self.report["timestamp_start"], "log_paths": self.report["log_paths"],
            "final_validation_completed": self.report["final_validation_completed"],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="Resume this experiment's last.ckpt; never selects a best model")
    args = parser.parse_args()
    start = time.perf_counter()
    r = initial_report()
    r.update(timestamp_start=now(), resume=args.resume)
    config = protocol()
    timing = {"prior": 0.0, "start": None, "finished": None}
    stage = "OTHER"
    torch = engine = logger = None
    memory_ready = False
    preserve_existing_report = FINAL.exists() or (LAST.exists() and not args.resume)
    code = 1
    try:
        os.chdir(ROOT)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "checkpoints").mkdir(exist_ok=True)
        stage = "CHECKPOINT"
        # Do not overwrite an already completed baseline or silently restart a partial run.
        if FINAL.exists():
            raise FileExistsError(f"Final checkpoint already exists: {FINAL}")
        if LAST.exists() and not args.resume:
            raise FileExistsError("last.ckpt exists; use --resume to continue this fixed-step experiment.")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        import torch
        import torchvision
        from importlib.metadata import version
        from lightning.pytorch import Callback, seed_everything
        from lightning.pytorch.callbacks import LearningRateMonitor
        from lightning.pytorch.loggers import CSVLogger
        from anomalib.callbacks import ModelCheckpoint
        from anomalib.data import MVTecAD2
        from anomalib.models import EfficientAd
        from anomalib.engine import Engine

        seed_everything(42, workers=True)
        versions = {"python_version": platform.python_version(), "torch_version": str(torch.__version__),
                    "torchvision_version": torchvision.__version__, "cuda_runtime": torch.version.cuda,
                    "anomalib_version": version("anomalib"), "lightning_version": version("lightning"),
                    "cudnn_deterministic": torch.backends.cudnn.deterministic,
                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "cuda_available": torch.cuda.is_available()}
        config.update(versions)
        r.update(versions)
        stage = "CUDA"
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; GPU 0 is required.")
        torch.cuda.set_device(0)
        props = torch.cuda.get_device_properties(0)
        r.update(gpu_name=props.name, gpu_total_memory_mib=props.total_memory / MIB)
        stage = "LOGGING"
        environment_snapshot({**versions, "gpu_name": props.name, "gpu_total_memory_mib": props.total_memory / MIB})
        stage = "ENVIRONMENT"
        r["b0_consistency_check"] = verify_b0_consistency(config, versions, props.name)
        config["b0_consistency_check"] = r["b0_consistency_check"]
        stage = "DATA"
        if not (Path(config["dataset_root"]) / "sheet_metal").is_dir():
            raise FileNotFoundError("Existing sheet_metal data required; downloading MVTec is prohibited.")
        datamodule = MVTecAD2(
            root="./datasets/MVTec_AD_2", category="sheet_metal", train_batch_size=1,
            eval_batch_size=1, num_workers=0, test_type="public",
        )
        datamodule.setup()
        for key, data in (("train_count", datamodule.train_data), ("validation_count", datamodule.val_data),
                          ("test_public_count", datamodule.test_public_data)):
            r[key] = len(data)
            if len(data) != config[key]:
                raise ValueError(f"{key}: expected {config[key]}, got {len(data)}; data unchanged.")
        stage = "MODEL_INIT"
        preprocessor, preprocessing = official_preprocessor(EfficientAd, 384)
        print("B0 preprocessing:", preprocessing["default_transform"], flush=True)
        print("384 preprocessing:", preprocessing["selected_transform"], flush=True)
        print("Only resize differs:", preprocessing["only_resize_changed"], flush=True)
        config["preprocessor_configuration"] = preprocessing
        r["preprocessor_configuration"] = preprocessing
        model = EfficientAd(model_size="small", imagenet_dir="./datasets/imagenette",
                            pre_processor=preprocessor, evaluator=False, visualizer=False, post_processor=False)
        config.update(parameter_count_total=sum(p.numel() for p in model.parameters()),
                      parameter_count_requires_grad=sum(p.numel() for p in model.parameters() if p.requires_grad),
                      optimizer_parameter_count=sum(p.numel() for p in model.model.student.parameters()) +
                                                sum(p.numel() for p in model.model.ae.parameters()),
                      learning_rate=model.lr, weight_decay=model.weight_decay)
        frozen_config = json.loads((B0_DIR / "config.json").read_text(encoding="utf-8"))
        for key in ("parameter_count_total", "optimizer_parameter_count", "learning_rate", "weight_decay"):
            if config[key] != frozen_config[key]:
                raise ValueError(f"Model/optimizer differs from B0: {key}")
        stage = "CHECKPOINT"
        if args.resume:
            # Only load a trusted local checkpoint generated by this script.
            saved = torch.load(LAST, map_location="cpu", weights_only=False)
            metadata = saved.get("phase08b", {})
            if metadata.get("protocol") != protocol():
                raise ValueError("Resume checkpoint protocol does not match this baseline.")
            step = int(saved["global_step"])
            if not 0 < step <= 70000 or not saved.get("optimizer_states") or not saved.get("lr_schedulers"):
                raise ValueError("Resume requires a complete Lightning checkpoint with step in 1..70000.")
            for key in ("torch_version", "anomalib_version", "lightning_version"):
                if metadata["config"].get(key) != config[key]:
                    raise ValueError(f"Resume framework mismatch: {key}")
            r.update(resume_checkpoint=str(LAST), resume_start_step=step,
                     timestamp_start=metadata["timestamp_start"], log_paths=metadata["log_paths"])
            timing["prior"] = float(metadata["training_elapsed_seconds"])
            del saved
        stage = "LOGGING"
        logger = CSVLogger(save_dir=str(OUT / "logs"), name="baseline", flush_logs_every_n_steps=100)
        r["log_paths"] = list(r["log_paths"]) + [str(Path(logger.log_dir) / "metrics.csv")]
        write_json(OUT / "config.json", config)
        logger.log_hyperparams(config)
        monitor = BaselineMonitor(r, config, timing)
        # Use anomalib's official subclass so Engine does not inject another checkpoint callback.
        recovery = ModelCheckpoint(dirpath=str(OUT / "checkpoints"), monitor=None, save_top_k=0,
                                   save_last=True, every_n_train_steps=1000, save_on_train_epoch_end=False,
                                   enable_version_counter=False)
        engine = Engine(max_steps=70000, max_epochs=1000, accelerator="gpu", devices=1,
                        precision="32-true", num_sanity_val_steps=0, limit_train_batches=1.0,
                        limit_val_batches=0, enable_checkpointing=True,
                        default_root_dir=str(OUT), logger=logger, log_every_n_steps=1,
                        callbacks=[monitor, LearningRateMonitor(logging_interval="step"), recovery])
        stage = "CUDA"
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        memory_ready = True
        torch.cuda.synchronize()
        stage = "TRAINING"
        timing["start"] = time.perf_counter()
        engine.fit(model=model, datamodule=datamodule, ckpt_path=str(LAST) if args.resume else None)
        torch.cuda.synchronize()
        r["session_training_elapsed_seconds"] = time.perf_counter() - timing["start"]
        timing["finished"] = timing["prior"] + r["session_training_elapsed_seconds"]
        r.update(completed_steps=int(engine.trainer.global_step), global_step=int(engine.trainer.global_step),
                 final_epoch=int(engine.trainer.current_epoch))
        if r["completed_steps"] != 70000:
            raise RuntimeError(f"Expected 70000 steps, got {r['completed_steps']}")
        stage = "LOGGING"
        logger.save()
        # A resume from step 70000 can produce no additional training rows.
        paths = [path for path in r["log_paths"] if Path(path).exists()]
        check_logs(paths, r)
        stage = "VALIDATION"
        validation_start = time.perf_counter()
        engine.trainer.limit_val_batches = 1.0
        engine.trainer.validate(model=model, datamodule=datamodule, ckpt_path=None, verbose=False)
        torch.cuda.synchronize()
        r["validation_elapsed_seconds"] = time.perf_counter() - validation_start
        if r["final_validation_batches"] != 19:
            raise RuntimeError(f"Final validation must consume 19 batches; got {r['final_validation_batches']}")
        quantiles = {key: float(value.detach().cpu().item()) for key, value in model.model.quantiles.items()}
        if not all(math.isfinite(value) for value in quantiles.values()):
            raise ValueError(f"Nonfinite final feature-map quantiles: {quantiles}")
        if quantiles["qb_st"] <= quantiles["qa_st"] or quantiles["qb_ae"] <= quantiles["qa_ae"]:
            raise ValueError(f"Degenerate final feature-map quantiles: {quantiles}")
        r.update(final_validation_completed=True, final_quantiles=quantiles)
        stage = "CHECKPOINT"
        engine.trainer.save_checkpoint(str(FINAL), weights_only=False)
        saved_final = torch.load(FINAL, map_location="cpu", weights_only=False)
        if (saved_final.get("global_step") != 70000 or not saved_final.get("optimizer_states")
                or not saved_final.get("lr_schedulers")
                or not saved_final.get("phase08b", {}).get("final_validation_completed")):
            raise ValueError("Final checkpoint is missing final-step, validation, or optimizer/scheduler state.")
        for key, value in quantiles.items():
            if float(saved_final["state_dict"][f"model.quantiles.{key}"].item()) != value:
                raise ValueError(f"Saved final quantile mismatch: {key}")
        del saved_final
        digest = hashlib.sha256()
        with FINAL.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * MIB), b""):
                digest.update(chunk)
        r.update(checkpoint_path=str(FINAL), checkpoint_size_mb=FINAL.stat().st_size / 1_000_000,
                 checkpoint_sha256=digest.hexdigest(), status="PASS")
        code = 0
    except (Exception, KeyboardInterrupt) as exc:
        if "cuda" in str(exc).lower() or "cudnn" in str(exc).lower():
            stage = "CUDA"
        r.update(status="FAIL", failure_stage=stage, error_message=f"{type(exc).__name__}: {exc}",
                 traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        if timing["start"] is not None:
            if r["session_training_elapsed_seconds"] is None:
                r["session_training_elapsed_seconds"] = time.perf_counter() - timing["start"]
            elapsed = timing["prior"] + r["session_training_elapsed_seconds"]
            r.update(training_elapsed_seconds=elapsed, training_elapsed_minutes=elapsed / 60,
                     training_elapsed_hours=elapsed / 3600)
        if engine is not None:
            try:
                r.update(completed_steps=int(engine.trainer.global_step), global_step=int(engine.trainer.global_step),
                         final_epoch=int(engine.trainer.current_epoch))
            except Exception as exc:
                r["warnings"].append(f"Trainer state unavailable: {exc}")
        if r["completed_steps"] and r["training_elapsed_seconds"] is not None:
            r["seconds_per_step"] = r["training_elapsed_seconds"] / r["completed_steps"]
        if r["training_elapsed_hours"] is not None:
            r["training_time_ratio_vs_b0"] = r["training_elapsed_hours"] / B0_HOURS
        if memory_ready:
            try:
                r.update(peak_gpu_allocated_mib=torch.cuda.max_memory_allocated() / MIB,
                         peak_gpu_reserved_mib=torch.cuda.max_memory_reserved() / MIB)
                r["training_vram_ratio_vs_b0"] = r["peak_gpu_allocated_mib"] / B0_VRAM_MIB
            except Exception as exc:
                traceback.print_exc()
                r["warnings"].append(f"VRAM reporting failed: {exc}")
                if code == 0:
                    r.update(status="FAIL", failure_stage="CUDA", error_message=str(exc))
                    code = 1
        if logger is not None:
            try:
                logger.save()
            except Exception as exc:
                traceback.print_exc()
                r["warnings"].append(f"CSV flush failed: {exc}")
                if code == 0:
                    r.update(status="FAIL", failure_stage="LOGGING", error_message=str(exc))
                    code = 1
        r.update(timestamp_end=now(), total_session_elapsed_seconds=time.perf_counter() - start)
        try:
            if preserve_existing_report and (OUT / "training_report.json").exists():
                print("Existing experiment report preserved; refused to overwrite prior results.", flush=True)
            else:
                write_reports(r)
        except Exception as exc:
            traceback.print_exc()
            r.update(status="FAIL", failure_stage=r["failure_stage"] or "LOGGING")
            r["error_message"] = (r["error_message"] or "") + f" Report writing failed: {exc}"
            code = 1
        print(summary(r), flush=True)
    return code


if __name__ == "__main__":
    # Use an importable module identity in checkpoints, never __main__.BaselineMonitor.
    if __package__:
        from .train_baseline384 import main as run_main
    else:
        from train_baseline384 import main as run_main
    sys.exit(run_main())

