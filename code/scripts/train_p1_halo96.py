"""Phase 10A P1 halo96 baseline. Run manually; never evaluates public test data.

python scripts/train_p1_halo96.py
python scripts/train_p1_halo96.py --resume
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
    from .train_baseline384 import BaselineMonitor as B1BaselineMonitor
except ImportError:
    from train_baseline384 import BaselineMonitor as B1BaselineMonitor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.p1_halo_dataset import P1HaloDataModule
from src.data.p1_halo_builder import DERIVED, read_rows, manifest_hash
AUDIT = ROOT / "experiments/phase09d_p1_halo_audit/p1_halo_audit.json"
OUT = ROOT / "experiments" / "phase10a_p1_halo96_seed42"
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
        "experiment_name": "P1_Halo96", "baseline_reference": "B0", "geometry_audit": "Phase09D",
        "dataset": "MVTec AD 2", "category": "sheet_metal", "dataset_root": str(DERIVED),
        "source_dataset_root": str(ROOT / "datasets/MVTec_AD_2"),
        "source_resolution": [1056,4224], "core_size": [1056,1056], "halo": 96,
        "context_size": [1248,1248], "model_input": [256,256], "input_size": [256,256],
        "tiles_per_source": 4, "train_count": 548, "validation_count": 76, "test_public_count": 456,
        "source_counts": {"train":137,"validation":19,"test_public":114}, "test_type":"public",
        "model":"EfficientAD-S", "model_size":"small", "steps":70000, "max_steps":70000, "max_epochs":1000,
        "batch":1, "batch_size":1, "num_workers":0, "seed":42,
        "precision":"FP32", "trainer_precision":"32-true", "accelerator":"gpu","devices":1,"gpu_index":0,
        "optimizer":"Adam","learning_rate":0.0001,"weight_decay":0.00001,
        "scheduler":"StepLR","scheduler_step_size":66500,"scheduler_gamma":0.1,
        "validation_during_fit":False,"final_full_validation":True,
        "evaluator":False,"visualizer":False,"post_processor":False,
        "pre_processor":"EfficientAD default 256x256; no custom PreProcessor",
        "early_stopping":False,"checkpoint_selection":"final step plus final validation",
        "training_core_crop":False,"padding":False,"pad_maps":True,
        "inference_geometry":"Restore256 map to1248, crop[96:1152,96:1152], concatenate4 cores; not executed in Phase10A",
        "sampling_relation":"548=137*4; uniform tile draw:4/548=1/137 source probability. Fixed70000 optimizer steps, not280000.",
        "geometry_facts_not_performance":{
            "halo_model_equivalent":19.6923,"strict_valid_border":18,"safety_margin":1.6923,
            "horizontal_detail_gain_vs_B0":3.3846,"uncovered_pixels":0,"duplicate_core_pixels":0,
            "gt_regions_covered":450,"gt_regions_total":450,"gt_blind_pixels":0,
            "p0_structural_seam":112,"p1_structural_seam":0},
        "difference_from_B0":"Input data geometry only: four reflected halo96 contexts per source; unchanged EfficientAD-S and optimizer."
    }


def verify_data(datamodule):
    """Read-only fixed manifest, provenance and official loader split checks."""
    audit=json.loads(AUDIT.read_text(encoding="utf-8"))
    build=json.loads((DERIVED/"dataset_report.json").read_text(encoding="utf-8"))
    digest=manifest_hash()
    if (audit.get("status")!="PASS" or not audit.get("formal_training_ready")
            or build.get("status")!="PASS" or build.get("manifest_sha256")!=digest
            or audit["dataset_report"]["manifest_sha256"]!=digest):
        raise ValueError("Frozen Phase09D build/geometry/smoke gate or manifest hash failed")
    rows=read_rows()
    for split,expected in protocol()["source_counts"].items():
        part=[a for a in rows if a["split"]==split]
        if len(part)!=expected*4 or len({a["source_image_id"] for a in part})!=expected:
            raise ValueError(f"Unexpected source/context counts: {split}")
        groups={}
        for a in part:
            groups.setdefault(a["source_image_id"],[]).append(a)
            if (int(a["halo"])!=96 or int(a["context_size"])!=1248 or int(a["core_size"])!=1056
                    or json.loads(a["core_offset_in_context"])!=[96,96]):
                raise ValueError("Manifest geometry differs from frozen P1")
            tile=int(a["tile_id"])
            if (int(a["core_x0"])!=tile*1056 or int(a["core_x1"])!=(tile+1)*1056
                    or int(a["core_y0"])!=0 or int(a["core_y1"])!=1056):
                raise ValueError("Core bounds changed")
            source=Path(a["source_image_path"]).resolve()
            source_root=(ROOT/"datasets/MVTec_AD_2/sheet_metal"/split).resolve()
            if not source.is_relative_to(source_root):
                raise ValueError("Cross-split source leakage")
            context=Path(a["context_path"]).resolve()
            if not context.is_relative_to((DERIVED/split/"images").resolve()) or not context.is_file():
                raise ValueError("Context missing or outside the fixed split")
            if split!="test_public" and (int(a["source_label"])!=0 or int(a["tile_is_anomalous"])!=0
                    or int(a["core_positive_pixels"])!=0 or source.parent.name!="good"):
                raise ValueError("Train/validation must contain only normal source images")
        for group in groups.values():
            if len(group)!=4 or {int(a["tile_id"]) for a in group}!={0,1,2,3} or len({a["source_image_path"] for a in group})!=1:
                raise ValueError("Each source must have exactly four unique contexts")
    if len(rows)!=1080 or len({a["context_path"] for a in rows})!=1080:
        raise ValueError("Unexpected/duplicate manifest rows")
    datamodule.setup()
    for split,data in (("train",datamodule.train_data),("validation",datamodule.val_data),("test_public",datamodule.test_data)):
        expected={str(Path(a["context_path"]).resolve()) for a in rows if a["split"]==split}
        if {str(Path(p).resolve()) for p in data.samples.image_path}!=expected:
            raise ValueError("DataModule split differs from manifest")
    return {"manifest_sha256":digest,"phase09d_audit_report":str(AUDIT),
            "phase09d_audit_sha256":hashlib.sha256(AUDIT.read_bytes()).hexdigest(),
            "source_counts":protocol()["source_counts"],"tile_counts":build["tile_counts"]}


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
    lines = ["=" * 40, "PHASE 10A - P1 HALO96 FORMAL TRAINING", "=" * 40,
             "Dataset: MVTec AD 2 / sheet_metal", "Model: EfficientAD-S", "Model input: 256x256",
             "Core:1056x1056; Halo:96; Context:1248x1248; Tiles/source:4",
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
             f"Inf detected: {r['loss_inf_detected']}",
             f"Total parameters: {r.get('total_parameters')}",f"Same parameters as B0: {r.get('same_parameters_as_b0')}",
             f"Resume used: {r['resume']}; from step: {r.get('resume_start_step')}", "", "STATUS:", r["status"]]
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
    for key in ("model_size","input_size","batch_size","num_workers","seed","max_steps","max_epochs",
                "optimizer","learning_rate","weight_decay","scheduler","scheduler_step_size","scheduler_gamma",
                "validation_during_fit","final_full_validation","evaluator","visualizer","post_processor","pre_processor",
                "early_stopping","checkpoint_selection"):
        if config[key]!=frozen[key]:raise ValueError(f"B0 model/training protocol differs: {key}")
    if frozen["precision"]!="32-true" or config["trainer_precision"]!="32-true":
        raise ValueError("FP32 protocol differs")
    changed={"input_data_geometry":config["difference_from_B0"]}
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


class BaselineMonitor(B1BaselineMonitor):
    """Reuse Phase08B pickle-safe callback, changing only experiment metadata."""
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        super().on_save_checkpoint(trainer, pl_module, checkpoint)
        metadata=checkpoint.pop("phase08b")
        metadata["protocol"]=protocol()
        checkpoint["phase10a"]=metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="Resume this experiment's last.ckpt; never selects a best model")
    parser.add_argument("--check-only", action="store_true", help="Validate data/model/environment/pickling and write READY report; no training")
    args = parser.parse_args()
    start = time.perf_counter()
    r = initial_report()
    r.update(timestamp_start=now(), resume=args.resume, resumed=args.resume, resume_from_step=0)
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
        (OUT / "logs").mkdir(exist_ok=True)
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
        datamodule=P1HaloDataModule()
        data_check=verify_data(datamodule)
        config.update(data_check);r.update(data_check)
        r.update(train_count=len(datamodule.train_data),validation_count=len(datamodule.val_data),
                 test_public_count=len(datamodule.test_data))
        environment_snapshot({**versions,"gpu_name":props.name,"derived_dataset":str(DERIVED),**data_check})
        stage="MODEL_INIT"
        print("Native context:1248x1248; Model input:256x256; Core retained at inference only:1056x1056",flush=True)
        model=EfficientAd(model_size="small",imagenet_dir="./datasets/imagenette",
                         evaluator=False,visualizer=False,post_processor=False)
        transform=model.pre_processor.transform
        from torchvision.transforms.v2 import Compose, Resize
        if not isinstance(transform,Compose) or len(transform.transforms)!=1:
            raise ValueError("Unexpected default preprocessing")
        resize=transform.transforms[0]
        if not isinstance(resize,Resize) or list(resize.size)!=[256,256] or not resize.antialias:
            raise ValueError("Default256 preprocessing changed")
        if model.model.ae.decoder.padding is not False or model.model.pad_maps is not True:
            raise ValueError("EfficientAD padding configuration changed")
        config["preprocessor_configuration"]=repr(transform)
        config.update(parameter_count_total=sum(p.numel() for p in model.parameters()),
                      parameter_count_requires_grad=sum(p.numel() for p in model.parameters() if p.requires_grad),
                      optimizer_parameter_count=sum(p.numel() for p in model.model.student.parameters()) +
                                                sum(p.numel() for p in model.model.ae.parameters()),
                      learning_rate=model.lr, weight_decay=model.weight_decay)
        frozen_config = json.loads((B0_DIR / "config.json").read_text(encoding="utf-8"))
        for key in ("parameter_count_total", "parameter_count_requires_grad", "optimizer_parameter_count", "learning_rate", "weight_decay"):
            if config[key] != frozen_config[key]:
                raise ValueError(f"Model/optimizer differs from B0: {key}")
        r.update(total_parameters=config["parameter_count_total"],trainable_parameters=config["parameter_count_requires_grad"],
                 optimizer_parameters=config["optimizer_parameter_count"],same_parameters_as_b0=True)
        stage = "CHECKPOINT"
        if args.resume:
            # Only load a trusted local checkpoint generated by this script.
            saved = torch.load(LAST, map_location="cpu", weights_only=False)
            metadata = saved.get("phase10a", {})
            if metadata.get("protocol") != protocol():
                raise ValueError("Resume checkpoint protocol does not match this baseline.")
            step = int(saved["global_step"])
            if not 0 < step <= 70000 or not saved.get("optimizer_states") or not saved.get("lr_schedulers") or not saved.get("state_dict") or not saved.get("callbacks"):
                raise ValueError("Resume requires a complete Lightning checkpoint with step in 1..70000.")
            for key in ("torch_version", "anomalib_version", "lightning_version", "manifest_sha256", "phase09d_audit_sha256"):
                if metadata["config"].get(key) != config[key]:
                    raise ValueError(f"Resume framework mismatch: {key}")
            r.update(resume_checkpoint=str(LAST), resume_start_step=step, resume_from_step=step,
                     timestamp_start=metadata["timestamp_start"], log_paths=metadata["log_paths"])
            timing["prior"] = float(metadata["training_elapsed_seconds"])
            del saved
        if args.check_only:
            import pickle
            monitor=BaselineMonitor(r,config,timing)
            restored=pickle.loads(pickle.dumps(monitor))
            if type(restored) is not BaselineMonitor or restored.state_key!="BaselineMonitor":
                raise ValueError("Callback pickle round trip failed")
            config["callback_pickle_check"]="PASS"
            write_json(OUT/"config.json",config)
            r.update(status="READY_NOT_STARTED",callback_pickle_check="PASS")
            code=0
            return code
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
        stage="CHECKPOINT"
        saved_last=torch.load(LAST,map_location="cpu",weights_only=False)
        if (saved_last.get("global_step")!=70000 or not saved_last.get("state_dict")
                or not saved_last.get("optimizer_states") or not saved_last.get("callbacks")
                or saved_last.get("phase10a",{}).get("protocol")!=protocol()):
            raise ValueError("last.ckpt failed full load/integrity check")
        r["last_checkpoint_loadable"]=True
        del saved_last
        stage = "VALIDATION"
        validation_start = time.perf_counter()
        engine.trainer.limit_val_batches = 1.0
        engine.trainer.validate(model=model, datamodule=datamodule, ckpt_path=None, verbose=False)
        torch.cuda.synchronize()
        r["validation_elapsed_seconds"] = time.perf_counter() - validation_start
        if r["final_validation_batches"] != 76:
            raise RuntimeError(f"Final validation must consume 76 batches; got {r['final_validation_batches']}")
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
                or not saved_final.get("phase10a", {}).get("final_validation_completed")):
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
        r["time_ratio_vs_B0"]=r["training_time_ratio_vs_b0"]
        r["vram_ratio_vs_B0"]=r["training_vram_ratio_vs_b0"]
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
        from .train_p1_halo96 import main as run_main
    else:
        from train_p1_halo96 import main as run_main
    sys.exit(run_main())

