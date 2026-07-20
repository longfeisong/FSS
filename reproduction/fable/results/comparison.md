# FABLE GELU block results

| Run | Queries | Chunks | LUT | Protocol ms | Wall ms | Comm GiB | Correctness |
|---|---:|---:|---:|---:|---:|---:|---|
| 20260717T171700-fable-gelu-block | 65536 | 16 × 4096 | 8→37 (padded 65536) | 75437 | 85000 | 18.042 | all-chunks-zero-error |

The report-ready workbook `FABLE_GELU替换前后对比.xlsx` contains the normalized
before/after table, metric definitions, raw values, reporting guidance and an
overhead-ratio chart. Regenerate it with
`conda run -n base python reproduction/fable/generate_gelu_comparison_excel.py`.
