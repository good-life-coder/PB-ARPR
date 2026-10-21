# Security audit

Status: PASS_LOCAL_RELEASE_SCREENING. The user explicitly authorized public GitHub distribution after the original review-package audit.

Scope: every candidate payload text/config/code file, plus archive member names. Allowlisted text formats only (Python, Markdown, CSV, JSON, YAML, text); no raw images, GT masks, RLE annotation files, checkpoints, banks, prediction arrays, binary caches, repository metadata or environment binaries. Only explicit aggregate Severstal FINAL_BLIND files were opened/copied. No dataset directories were opened for contents. Unrelated research paths were excluded.

Local project prefixes and machine user/environment paths are removed in release copies; transformations and both hashes are recorded. Credential patterns, private keys, access tokens, personal path patterns and unrelated-research markers are screened. Three email-pattern hits in weighted_bootstrap_exact.py are false positives from Python matrix multiplication `m@np.asarray`, manually inspected; no email address is present there. No other unresolved screen findings.

Original source code was not changed. Publication copies retain algorithm AST excluding string constants; path/string changes are separately documented. JSON projections retain scientific aggregates while omitting nonaggregate identity/cache fields. Six identical non-code files are deduplicated by SHA256; two empty package initializers remain for import compatibility.

No third-party datasets or weights are distributed. Dataset official acquisition/terms were checked; Severstal terms are not represented as unrestricted research permission. No project-wide license was invented; public distribution is author-authorized; no open-source license grant is inferred. Secret-pattern scanning is not a mathematical guarantee of absence, and this audit is not a full legal ownership certification.

The original audit preceded all uploads. Public GitHub release is now authorized; Zenodo publication is not part of this task. See the separate GitHub publication report for actual remote status.
