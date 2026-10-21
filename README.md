# PB-ARPR: Reproducibility Materials for Industrial Anomaly Localization

Public research reproducibility materials made available by author authorization. Author: Bin Liang.

Associated manuscript: **Input Geometry Under Pixel-Budget Constraints for High-Resolution Industrial Anomaly Localization**. Intended venue: Journal of Electronic Imaging (JEI). This repository does not claim acceptance or publication, and no paper DOI is assigned here.

No data or model weights are distributed. The original review ZIP is preserved unchanged as an archival artifact; its confidential-review wording records the earlier distribution context. The author subsequently authorized this public release. Full end-to-end reproduction has not been independently verified for every experiment.
PB-ARPR investigates pixel-budgeted, aspect-ratio-preserving network inputs, with architecture feasibility fallback, for industrial anomaly localization. It does not introduce a new detector backbone.

## Start here: verified, data-free checks

From this package directory, using Python 3.10:
```text
python statistics/verify_aggregates.py
python statistics/verify_synthetic_metrics.py
```
The first recomputes five-seed means, sample SD and paired differences, checks KSDD2 per-seed aggregate records and Severstal aggregate deltas. It does not recompute AUROC from predictions. The second checks the original Exact AUPRO integration with small synthetic inputs (numpy required). See REPRODUCIBILITY_AUDIT.md for actual execution results and boundaries.

## Install in a separate environment

Do not change an existing research environment. The recorded environment is Python 3.10.21, torch 2.9.1+cu126, torchvision 0.24.1+cu126, anomalib 2.6.1, lightning 2.6.6. Install the matching CUDA wheels using the official PyTorch installation instructions, then `python -m pip install -r requirements.txt`. Requirements are observed versions, not a newly tested clean-room lock. Offline wheel availability and clean installation are UNVERIFIED. Historical environment snapshots are under experiments; current package inventory is configs/current_environment.json. See ENVIRONMENT.md.

## Data and weights

Use only the official acquisition routes in DATASETS.md and accept their terms. No private test image/label/identity or per-image blind predictions are provided. Source code expects a project layout containing `datasets/MVTec_AD_2/sheet_metal`, a KSDD2 source discovered by its loader, and the original Severstal annotated-training organization. ImageNette is an external auxiliary training dependency. Models are identified by hashes in frozen reports; see docs/WEIGHTS.md. No data acquisition, training, model loading or inference was executed for this package.

## Input geometry

Run data-free geometry selection from `code`:
```text
python -c "from src.common.pbarpr_geometry import select_rect; print(select_rect(637/229))"
```
`select_rect` accepts native height/width. `src/common/architecture_constrained_geometry.py` contains raw-first selection, candidate enumeration and model feasibility checking; its `select(ratio, checker)` preserves the frozen fallback. Synthetic model compatibility checks require weights/dependencies and are not run here. Historical sheet_metal 256x1024 was manually fixed; KSDD2 864x320 reused an earlier selector involving test-size metadata; Severstal 256x1152 used architecture fallback. The later normal-only audit is retrospective, not historical training provenance.

## Historical experiment entry points (not executed by this package)

The `code` tree preserves the original `scripts`, `src` and submission-audit dependencies. Helper files from earlier second-paper experiments are included only where imports require them; their presence is not a claim of formal evidence. Python syntax and dependency closure are checked without importing training entry points.

| Task | Source entry |
|---|---|
| sheet_metal PB-ARPR | code/scripts/train_rectangular_efficientad_256x1024.py |
| Square512 seed42 | code/scripts/train_baseline512.py |
| Seeds43–46 Square/PB | code/scripts/phase24b_train_square.py; code/scripts/phase24b_train_rect.py (PHASE24B_SEED=43..46) |
| Tile4 | code/scripts/train_tile4_70k.py |
| PatchCore fit / inference | code/scripts/run_patchcore_p2.py; code/scripts/patchcore_p2_evaluation.py |
| PatchCore equal-pixel comparison | code/scripts/phase24a_p2_formal.py |
| KSDD2 | code/scripts/train_ksdd2_70k_std_rect.py; code/scripts/train_ksdd2_repeat_seeds.py |
| Severstal normal training | code/scripts/train_severstal_70k_std_rect.py |
| Multi-budget | code/scripts/phase24c_r1_audit.py; code/scripts/phase24c_r1_train_mid.py |
| Native map restore / pixel AUROC | code/scripts/evaluate_rectangular_efficientad_public.py; code/src/evaluation/rectangular_formal_evaluation.py |
| Exact AUPRO | code/src/evaluation/exact_aupro.py |
| Paired image bootstrap | code/src/evaluation/weighted_bootstrap_exact.py; code/scripts/run_phase21b_b.py |
| Unified image-score audit | code/experiments_new/pbarpr_submission_audit_r1/run_audit.py |

These are historical, gated entry points, not portable one-command launches. They derive ROOT from their location, expect prior-stage artifacts and may refuse existing completed runs. To reproduce, use a **new isolated workspace**, place `code/scripts` and `code/src` at its root, obtain authorized data/weights, and reconcile frozen prerequisite paths/configuration against included experiments. Do not launch against the original project or delete completion gates. The package does not silently rewrite gates or reconstruct withheld split identities. Full training, cached bootstrap and end-to-end evaluation remain **UNVERIFIED** and may need author-assisted preparation. Severstal FINAL_BLIND is aggregate-only; its inference entry point is intentionally omitted.

## Exact AUPRO and scoring

`exact_aupro.integrate(background, scores, weights, np)` consumes sorted negative-pixel scores and positive scores weighted by inverse region size / total connected regions. It integrates all score ranks, including ties and the partial FPR-boundary segment, at 0.05 and 0.30; it is not a sampled threshold grid. `components` uses 8-connectivity. `native` restores maps and requires externally obtained authorized masks. No pixel results can be independently regenerated without these assets. Bootstrap resamples images, not independent pixels.

The current five-seed image score is max of the native-resolution anomaly map. Unified Square512=62.583333±1.056935%, PB-ARPR=82.101852±3.290719%, paired gain=19.518519±4.227767 pp (sample SD, ddof=1). This does not remove historical antialias confounding.

## Results and traceability

`results_summary.csv` is a long-form ledger, preserving table cell strings; `results_source_manifest.csv` maps copied files to source hashes and transformations; `results/paper_table_value_sources.csv` maps historical table values to original source keys. `results/historical_tables` preserves six main and two supplementary tables. These predate the unified audit: Square512 image AUROC must be read with the newer `experiments_new/pbarpr_submission_audit_r1/image_auroc_unified.csv`, not silently replaced. Phase24B, Phase24C-R1 and PatchCore geometry reports supplement the older tables. Actual submitted manuscript/table numbering was not supplied and is UNVERIFIED.

Directly checkable: supplied aggregate arithmetic, frozen configurations, source correspondence, selected training logs, hash integrity, synthetic metric behavior. Requiring full assets/new training: exact checkpoint regeneration, all native pixel metrics, population bootstrap from maps, original latency/GPU measurements and private blind evaluation. No such reruns are claimed.

Read KNOWN_LIMITATIONS.md before interpreting results. The ZIP is review material, not a certification of causal isolation or clean-room numerical equivalence.

Exact duplicate evidence files are stored once. Before any historical runtime preparation in a new extracted workspace, `python statistics/restore_aliases.py` restores the six documented non-code aliases. Empty Python package initializers remain in place. Verify the original archive manifest before adding aliases.
