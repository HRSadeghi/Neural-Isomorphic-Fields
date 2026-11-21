# configs

`experiments.json` lists the 43 training runs of the paper with their exact command-line settings, in the order in
which `run_experiments.py` starts them:

* `kind: full` runs `train.py --name <name> <args>` (a complete NIF model),
* `kind: probe` runs `train_operator.py --name <name> <args>` (one operator alone),
* `test: lengths` marks the runs of the length study (N = 50, 100, 200), which are tested by `analysis/lengths.py`
  instead of `evaluate.py`.

The settings that a run actually used, defaults included, are stored in `runs/<name>/config.json`.

| Group | Runs | Paper |
|:---|:---|:---|
| main model | `main_s0`, `main_s1`, `main_s2` (seeds 0, 1, 2) | Table 4, Figures 5, 6 |
| ablations of the full model | `full_paper_losses`, `full_no_hints`, `full_no_targeted`, `full_no_length_mix`, `full_dense_ops` | Figure 8, Table A.3 |
| AE-only | `aeonly_rec` (encoder trained with ℓ_rec only), `aeonly` (that encoder frozen, the rest trained with the full objective) | Table 5, Table A.3 |
| embedding size | `emb_c2`, `emb_c4`, `emb_c6`, `emb_c16` | Figures 8, 9, Tables A.3, A.4 |
| number length × dimension | `len_n{50,100,200}_c{2,4,6,10}` | Figure 9, Table A.4 |
| operators alone | `abl_x_*`, `abl_add_*`, `depth_x_*` | Figure 7, Table A.2 |
