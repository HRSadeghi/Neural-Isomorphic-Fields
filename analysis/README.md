# analysis

Analyses of trained models. They read the weights from `checkpoints/` and write to `runs/<run>/` or
`results/analysis/`. Except for `measure_cost.py` they run on the CPU; `lengths.py` and `geometry.py` also
take `--device cuda`.

| Script | What it computes | Output |
|:---|:---|:---|
| `lengths.py` | exact match by number length, for any N | `runs/<run>/lengths.json`, `lengths_items.json.gz` |
| `parts.py` | joins those outcomes with the integer and fractional length of each test number (no model needed) | `runs/<run>/lengths_parts.json` |
| `geometry.py` | the digit code of one slot, defects and decoding radii of the selected model `main_s0` | `results/analysis/model_analysis.json` |
| `extra.py` | inverse test split by the slot window, kinds of reconstruction errors, the digit code of all models | `results/analysis/extra.json` |
| `digit_code.py` | Fourier analysis of the digit code; reconstruction with the true length beyond the training lengths | `results/analysis/extra.json` |
| `measure_cost.py` | time per number and peak GPU memory of each part of the model | one JSON line (`results/analysis/cost.jsonl`) |

```bash
python analysis/lengths.py                          # three seeds and two ablations
python analysis/lengths.py --runs len_n200_c4 --device cuda
python analysis/parts.py
python analysis/geometry.py
python analysis/extra.py --runs main_s0,main_s1,main_s2,emb_c4
NIF_N_DIGITS=50 python analysis/extra.py --runs len_n50_c4,len_n50_c10
python analysis/digit_code.py --runs main_s0,main_s1,main_s2
python analysis/measure_cost.py --ckpt checkpoints/main_s0/best.pt --digits 25
```

`extra.py` and `digit_code.py` need all runs of one call to have the same N; set `NIF_N_DIGITS` for N other than 25
(see the main README for Windows). Their logs go to `logs/`.

## Defects, decoding radii and Lipschitz constants (Table A.5)

Quantities of Section 3.6 on 2000 test tuples per test (length-free decoding, mean over the seeds; EM: exact match on
these tuples). Defect: median / 90th percentile; ρ̂: median estimated decoding radius (50 random perturbations of norm
δ_0 = 6, length head included in the margin); in bound: tuples for which Proposition 3.24(a), respectively the bounds of
Proposition 3.27 weighted with L̂, give the correct result for the estimated radius; wrong ball: the result lies in the
decoding ball of a wrong number; L̂: empirical Lipschitz constant. The radii and these shares are estimates, not
guarantees. Rows c = 2 … 16: single runs, values for + / ×.

| Test | EM (%) | Defect | ρ̂ | In bound (%) | Wrong ball (%) | L̂ |
|:---|---:|---:|---:|---:|---:|---:|
| +, closure | 100.00 | 0.73 / 1.10 | 2.88 | 99.8 | 0.0 | 2.52 |
| ×, closure | 99.98 | 0.83 / 1.08 | 2.80 | 99.9 | 0.0 | 3.94 |
| Associativity of + | 100.00 | 1.49 / 2.07 | 2.80 | 42.9 | – | – |
| Distributivity | 99.92 | 1.66 / 2.07 | 2.67 | 1.1 | – | – |
| c = 2: closure of + / × | 94.75 / 99.20 | 1.35 / 2.18 | 1.33 / 1.34 | 47.8 / 12.4 | 0.7 / 0.0 | 2.71 / 10.49 |
| c = 4: closure of + / × | 100.00 / 100.00 | 0.48 / 0.34 | 1.52 / 1.51 | 99.2 / 100.0 | 0.0 / 0.0 | 2.22 / 4.12 |
| c = 6: closure of + / × | 99.65 / 100.00 | 1.52 / 1.78 | 2.16 / 2.05 | 87.5 / 72.0 | 0.2 / 0.0 | 2.24 / 3.80 |
| c = 16: closure of + / × | 100.00 / 100.00 | 0.88 / 1.29 | 3.35 / 3.18 | 99.2 / 99.5 | 0.0 / 0.0 | 3.96 / 4.03 |

These numbers come from `test_results.json` of each run (`geometry` entry, written by `evaluate.py`).

## Other results quoted in the paper

* Inverse of the units of D_25 (`extra.py`): for the 3790 units whose digit products fit the slot window,
  x × (1/x) is correct in 99.77% of the cases; for the other 648 units the product is cut off at
  the window (0.00% correct), and exact arithmetic on the 52 places would reach only 85.40%.
  With N = 50 and 100 the test reaches 100.00% for each of the 6 models,
  and between 99.19% and 100.00% for the 3
  models with N = 200.
* Digit code (`digit_code.py`, Figure 10a, b): the code of a slot is the same at every exponent (cosine similarity at
  least 0.99), the other digits account for at most 0.06% of the variance between digits,
  and its effective dimension is 2.6–4.1 of the 10 channels. The two lowest harmonics carry
  87–95% of the variance between the digit means of the three seeds; in `main_s0` the
  first harmonic alone carries 84%. With c = 2 the digit means lie on a line (effective dimension
  1.0); with c = 2 or perceptron AC layers the code differs between exponents (cosine similarity
  0.53 and 0.57).
* Reconstruction with the true length (`digit_code.py`, Section 5.3.2): for numbers with a longer part, giving the
  decoder the true length raises reconstruction from 52.4% to 98.1% (N = 100)
  and from 4.7% to 76.4% (N = 200).

## Inference cost (Table A.1)

`main_s0`, numbers with 25 digits, batch 512, on a partition of an NVIDIA H200 GPU
(`results/analysis/cost.jsonl`):

| Part | Time per number (ms) | Peak GPU memory per batch (MB) |
|:---|---:|---:|
| Encoder φ | 0.10 | 324 |
| Addition | 0.08 | 985 |
| Multiplication | 0.41 | 1701 |
| Comparator | 0.02 | 429 |
| Decoder (length-free) | 0.46 | 258 |
