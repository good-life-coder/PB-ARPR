# Known limitations and negative results

- Historical sheet_metal dimensions were manually fixed; later selection code must not be presented as already executed before those runs.
- KSDD2 historical ratio statistics included official-test image dimensions. The package retains this limitation; it does not repartition data or claim a normal-only historical protocol.
- Severstal used an architecture-constrained fallback (256x1152); the historical population used for geometry was not restricted to normal training examples. DEV normal samples were reused for quantile calibration, as stated by the frozen protocol.
- Square and PB preprocessing differ in antialias behavior. Unifying image-score calculation does not eliminate these confounders.
- Historical Square512 seed42 image AUROC 61.018519% was revised to native-map-max 61.064815%; seed46 also changed. Other detector scoring schemes were not globally reaudited by this package.
- The initial 256 baseline evaluation report records FAIL_AUPRO; use the separate Phase07C-R1 exact-AUPRO report for those metrics. Preserve rather than hide that repair history.
- Severstal FINAL_BLIND image AUROC decreases by 5.648731 pp despite pixel/localization gains. Sheet_metal pixel-AUROC bootstrap intervals for some comparisons cross zero. LOW pixel-budget non-square candidates are infeasible. These negative results remain included.
- Sheet_metal is one MVTec AD 2 category. Severstal is single-seed, unlike five-seed KSDD2 and sheet_metal geometry comparisons. SD across seeds is not a confidence interval.
- Tables are historical material candidates, not a verified copy of the final JEI manuscript. Full paper coverage cannot be certified without the submitted table list; no manuscript was modified.
- No original images, masks, private split identities, third-party model weights, trained checkpoints or prediction caches are distributed. Thus aggregate arithmetic is reproducible now; full experiments are not plug-and-play.
- Static Python import closure is not proof of dynamic file/runtime dependency completeness. Some historical entry points require previous-stage gates. Full execution is UNVERIFIED, with missing-runtime references listed in docs/runtime_references.csv.
- Author-created files have no discovered project-wide license. The author authorized public distribution; no open-source license or third-party redistribution right is invented. Third-party dependencies retain their own licenses.
