# results/csv

| File | Contents | Paper |
|:---|:---|:---|
| `main_results.csv` | every test of every full-model run, both decoding modes (n, EM, accuracy) | Table 4, Figure 8 |
| `ablation_operator.csv` | operators trained alone: EM, EM for 21–25 digits, steps, stage accuracies | Table A.2 |
| `ablation_operator_curves.csv` | their validation curves | Figure 7 |
| `ablation_full.csv` | ablations of the full model with the p-values of McNemar's tests | Table A.3 |
| `length_dimension.csv` | exact match by the shape of the number, per run | Table A.4, Figure 9 |
| `geometry.csv` | defects, decoding radii, shares in bound, Lipschitz constants | Table A.5 |
| `training_curves_mean.csv` | validation exact match of the three seeds (mean, min, max) | Figure 6 |
| `training_metrics.csv`, `validation_evals.csv` | `metrics.jsonl` and `evals.jsonl` of every run in one table | – |
| `run_summary.csv` | per run: last step, hours, finished | – |
| `reader.csv` | the reading experiment | Table 6 |
| `llm.csv` | base model and model trained with NIF, with McNemar p-values | Table 7 |
| `llm_causal.csv` | interventions on the trained language model | Figure 12 |
