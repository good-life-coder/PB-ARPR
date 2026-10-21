# Reproducibility audit

Status: LOCAL_STATIC_AND_AGGREGATE_CHECKS_PASS; FULL_REPRODUCTION_UNVERIFIED.

Executed: syntax compilation without bytecode for all packaged Python; JSON decoding; static src/scripts import closure; source SHA recheck; non-string AST equivalence for release code; five-seed aggregate arithmetic; KSDD2 ten per-seed aggregate comparisons; Severstal aggregate deltas; synthetic Exact AUPRO perfect/inverse/tied scores.

263 table source-key numeric checks match. 54 entries remain UNVERIFIED for generic automatic source-key checks (text, derived formatting, omitted identity-bearing sources). These are individually listed in docs/table_source_checks.csv. No detected numeric mismatch. The eight table CSVs preserve historical values; current unified Image AUROC is explicitly separate. Supplied table collection coverage is checked; actual JEI submitted-table coverage is UNVERIFIED because the final manuscript was not supplied.

No training, network inference, dataset reading, private label reading, bootstrap-from-maps rerun, checkpoint binary validation or clean-environment installation was executed. Full steps depending on authorized data, frozen split identities, weights and prior-stage gates are UNVERIFIED. Historical checkpoint SHA values are recorded evidence, not freshly verified binaries. Runtime string references are listed in docs/runtime_references.csv; missing prerequisites are not silently invented.

Source trees outside this new package were read only; all selected source hashes remain unchanged. This is a static dependency/provenance package with executable aggregate checks, not a turnkey trained-model distribution.
