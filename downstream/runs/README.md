# downstream/runs

Outputs of the runs of the paper (written by `downstream/train.py`, read by `downstream/make_tables.py`).

| Folder | Run |
|:---|:---|
| `base_model/` | Qwen3.5-0.8B as released |
| `operator_layers/` | trained with the NIF operator layers for 6000 steps; final test and interventions |
| `operator_layers_10000/` | the same run continued to 10,000 steps |

| File | Contents |
|:---|:---|
| `config.json` | every setting of the run |
| `metrics.jsonl` | training losses every 25 steps |
| `evals.jsonl`, `cells_steps.jsonl` | the complete test every 500 steps, by test cell (Figure 11c) |
| `results.json` | the final test, and for `operator_layers` the interventions |
| `items.jsonl.gz` | every test problem with the output of the model and whether it is correct |
| `causal_items.jsonl.gz` | outcomes with the operator layers switched off and with the vectors of other numbers |
| `DONE` | written at the end |

The weights of `operator_layers` after 6000 and 10,000 steps are in `checkpoints/language_model/` (see
[checkpoints/](../../checkpoints/)); the results are in [downstream/README.md](../README.md).
