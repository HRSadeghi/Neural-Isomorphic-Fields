# runs

One folder per training run of the paper (the settings are in [configs/experiments.json](../configs/experiments.json)).
The weights are not here but in `checkpoints/<run>/best.pt` (see [checkpoints/](../checkpoints/)).

Files of a run folder:

| File | Contents |
|:---|:---|
| `config.json` | every setting of the run |
| `metrics.jsonl` | every 50 steps: each loss term, gradient norm, learning rate, time, GPU memory |
| `evals.jsonl` | every validation (500 steps; 250 for the operators alone): exact match on held-out tuples, overall and by length |
| `test_results.json` | the tests of the paper in both decoding modes (`N25_length_free`, `N25_oracle`), written by `evaluate.py` |
| `test_results.items.json.gz` | the 0/1 outcome of every test item (McNemar's tests between models) |
| `lengths.json`, `lengths_items.json.gz`, `lengths_parts.json` | exact match by number length and by the shape of the number (analysis/lengths.py, analysis/parts.py) |
| `DONE` | the last training step |

All tables below are built by `python paper/make_tables.py` from these files (also as CSV in
[results/csv/](../results/csv/)); `paper/generated/` holds the LaTeX versions.

## Main model (Table 4)

`main_s0`, `main_s1`, `main_s2`: training stopped after 8,000, 9,000 and 8,000
steps; the checkpoint with the best validation score is kept. The table is in the [README of the repository](../README.md#main-result).

## Comparison protocol (Table 5)

The protocol of the comparison with other number embeddings: operands of at most N digits, the decoder is given the
length of the result. NIF: mean ± SD over the three seeds; AE-only (`aeonly`): an encoder trained with ℓ_rec only and
then frozen, the decoder, operators and comparator trained with the full objective, one run. For the order the value
is the share of pairs with the correct relation. The rows of LUNA, FoNE and xVal are given in Table 5 of the paper.

|  | Model | N = 10 Acc. | N = 10 EM | N = 20 Acc. | N = 20 EM | N = 30 Acc. | N = 30 EM |
|:---|:---|---:|---:|---:|---:|---:|---:|
| Addition | NIF | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 99.99 ± 0.01 |
|  | AE-only | 100.00 | 99.97 | 99.99 | 99.95 | 100.00 | 99.99 |
| Multiplication | NIF | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.01 | 100.00 ± 0.00 | 99.98 ± 0.01 |
|  | AE-only | 84.76 | 43.66 | 75.60 | 29.65 | 75.18 | 29.00 |
| Order | NIF | 99.74 ± 0.09 | – | 99.78 ± 0.07 | – | 99.79 ± 0.08 | – |
|  | AE-only | 99.46 | – | 99.34 | – | 99.60 | – |

## Operators trained alone (Figure 7, Table A.2)

Each operator is trained in isolation until convergence (at most 8000 steps) on fixed one-hot digit slots, without
encoder and decoder, and a linear head reads the digits of the result. EM: exact match (%) on 4096 held-out pairs of
the sampler, overall and for results with 21–25 digits, at the best validation; steps: training steps until
convergence; first ≥ 99%: first validation (every 250 steps) with EM ≥ 99% (–: never); the last three columns are the
stage accuracies of the staged product (operand digits, (L_k, H_k) within 0.5, and T_k).

| Operator | EM | EM, 21–25 digits | Steps | First ≥ 99% | Digits | (L_k, H_k) | T_k |
|:---|---:|---:|---:|---:|---:|---:|---:|
| ×: staged operator (Section 3.3.3) | 99.95 | 100.00 | 3250 | 2500 | 100.00 | 100.00 | 99.96 |
| &emsp;without hint supervision | 29.39 | 0.00 | 8000 | – | 0.00 | 1.69 | 8.54 |
| &emsp;without least-squares fit of the counts | 99.98 | 98.18 | 5750 | 5250 | 100.00 | 100.00 | 100.00 |
| &emsp;random instead of decimal frequencies | 99.98 | 100.00 | 5500 | 3500 | 100.00 | 100.00 | 100.00 |
| &emsp;without periodic features | 99.88 | 98.18 | 8000 | 6250 | 100.00 | 100.00 | 99.86 |
| &emsp;soft instead of one-hot digit counts | 99.98 | 100.00 | 3000 | 2250 | 100.00 | 100.00 | 99.98 |
| &emsp;bfloat16 instead of float32 | 99.95 | 98.18 | 4000 | 2000 | 100.00 | 100.00 | 100.00 |
| ×: generic AC layers with a place-coupled bilinear branch, k = 6 | 32.49 | 0.00 | 8000 | – | – | – | – |
| ×: perceptron AC layers, f(x) = σ(Wx+b), k = 4 | 10.52 | 0.00 | 8000 | – | – | – | – |
| +: slot-Transformer AC layers (Section 3.3), k = 6 | 100.00 | 100.00 | 3250 | 1750 | – | – | – |
| +: perceptron AC layers, f(x) = σ(Wx+b), k = 4 | 20.41 | 0.00 | 8000 | – | – | – | – |
| ×: perceptron AC layers, k = 1 | 9.69 | 0.00 | 1750 | – | – | – | – |
| ×: perceptron AC layers, k = 2 | 10.11 | 0.00 | 6500 | – | – | – | – |
| ×: perceptron AC layers, k = 6 | 10.13 | 0.00 | 4750 | – | – | – | – |
| ×: generic AC layers with the bilinear branch, k = 2 | 29.05 | 0.00 | 8000 | – | – | – | – |
| ×: generic AC layers with the bilinear branch, k = 4 | 37.23 | 0.00 | 8000 | – | – | – | – |
| ×: generic AC layers with the bilinear branch, k = 8 | 43.14 | 0.00 | 8000 | – | – | – | – |

Run names, from top to bottom: `abl_x_full`, `abl_x_nohints`, `abl_x_norefit`, `abl_x_fourier_random`,
`abl_x_fourier_none`, `abl_x_softcounts`, `abl_x_bf16`, `abl_x_coupled`, `abl_x_dense`, `abl_add_slot`,
`abl_add_dense`, `depth_x_dense_k1`, `depth_x_dense_k2`, `depth_x_dense_k6`, `depth_x_coupled_k2`,
`depth_x_coupled_k4`, `depth_x_coupled_k8`.

## Ablations of the full model (Figure 8, Table A.3)

Exact match (%, length-free decoding) on the test sets. Each variant changes one part of NIF and is trained with the
same budget and seed 0. `*`: different from the seed-0 run (McNemar's test, Holm–Bonferroni corrected p < 0.05 within
the table, all columns but the last). Rec.: reconstruction, Cl.: closure, Inv.: inverse, Ass.: associativity, Distr.:
distributivity, O1: compatibility of the order with +, last column: chains of 5 sums (500 chains).

| Variant | Rec. | Cl. + | Cl. × | Inv. + | Inv. × | Ass. + | Ass. × | Distr. | O1 | +, M = 5 |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NIF (all terms) | 100.00 | 100.00 | 99.98 | 100.00 | 85.22 | 99.98 | 99.94 | 99.92 | 99.44 | 99.87 |
| NIF, seed 0 (reference of the tests) | 99.99 | 99.99 | 99.97 | 99.99 | 84.86 | 99.99 | 99.95 | 99.88 | 99.57 | 100.00 |
| only ℓ_rec, ℓ_iso, ℓ_ord | 99.92 | 99.78\* | 22.47\* | 88.47\* | 0.72\* | 99.61\* | 5.12\* | 11.61\* | 92.71\* | 100.00 |
| without hint supervision | 99.92 | 99.80\* | 10.00\* | 100.00 | 17.44\* | 99.62\* | 3.78\* | 2.37\* | 99.74 | 99.80 |
| without targeted pairs and terms on operator outputs | 100.00 | 100.00 | 99.85 | 99.96 | 27.17\* | 99.88 | 99.70\* | 99.55\* | 86.43\* | 100.00 |
| training numbers only from the sampler (no uniform lengths) | 99.99 | 100.00 | 100.00 | 100.00 | 85.40\* | 99.99 | 99.98 | 99.97 | 99.76 | 100.00 |
| perceptron AC layers in both operators | 99.71\* | 4.40\* | 3.04\* | 100.00 | 1.80\* | 0.20\* | 3.34\* | 1.22\* | 99.23\* | 0.00 |
| AE-only encoder (trained with ℓ_rec only, then frozen) | 99.98 | 99.93 | 29.02\* | 100.00 | 31.14\* | 99.56\* | 5.96\* | 12.08\* | 98.84\* | 99.60 |
| embedding size c = 2 (d = 104) | 99.99 | 95.22\* | 99.01\* | 100.00 | 83.82\* | 15.90\* | 3.45\* | 10.80\* | 99.56 | 1.60 |
| embedding size c = 4 (d = 208) | 100.00 | 100.00 | 100.00 | 100.00 | 85.40\* | 99.73\* | 100.00 | 99.80 | 99.68 | 96.60 |
| embedding size c = 6 (d = 312) | 99.85\* | 99.74\* | 99.98 | 100.00 | 85.40\* | 98.76\* | 99.74\* | 98.30\* | 99.30 | 97.40 |
| embedding size c = 16 (d = 832) | 100.00 | 100.00 | 100.00 | 100.00 | 85.40\* | 99.98 | 100.00 | 99.99 | 99.73 | 100.00 |

Runs: `full_paper_losses`, `full_no_hints`, `full_no_targeted`, `full_no_length_mix`, `full_dense_ops`, `aeonly`,
`emb_c2`, `emb_c4`, `emb_c6`, `emb_c16`. The remaining tests of Figure 8 are in each run's `test_results.json`.

## Number length and embedding dimension (Figure 9, Table A.4)

Exact match (%, length-free decoding) by the shape of the number that the decoder has to produce (the number itself
for reconstruction, the result for the closure tests). "Within": integer and fractional part within the training
lengths (at most 12 digits each for N = 25, ⌊N/2⌋ otherwise) and at most N digits in total; "longer part": a longer
part, at most N digits; "> N digits": outside D_N. N = 25, c = 10: mean of the three seeds; all other rows: one run.
The runs with N = 200 and c ≥ 4 were trained for 10,300–11,700 steps.

| N | c | d | Within: Rec. | Within: + | Within: × | Longer part: Rec. | Longer part: + | Longer part: × | > N digits: Rec. | > N digits: + | > N digits: × |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 2 | 104 | 100 | 91.4 | 98.7 | 93.2 | 73.0 | 78.9 | 34.0 | 25.9 | 23.7 |
|  | 4 | 208 | 100 | 100 | 100 | 98.5 | 98.5 | 99.93 | 64.5 | 72.4 | 78.6 |
|  | 6 | 312 | 99.94 | 99.69 | 99.94 | 98.3 | 98.3 | 98.1 | 51.9 | 59.5 | 64.0 |
|  | 10 | 520 | 100 | 100 | 99.99 | 98.1 | 98.6 | 99.50 | 53.5 | 60.5 | 63.8 |
|  | 16 | 832 | 100 | 100 | 100 | 99.26 | 99.30 | 99.86 | 72.0 | 75.5 | 79.8 |
| &emsp;c = 10, no uniform lengths |  |  | 100 | 100 | 100 | 97.1 | 98.9 | 99.36 | 43.3 | 50.8 | 55.9 |
| &emsp;c = 10, no targeted pairs |  |  | 100 | 100 | 99.94 | 92.1 | 94.6 | 98.1 | 49.3 | 54.1 | 65.3 |
| 50 | 2 | 204 | 100 | 85.9 | 26.5 | 22.7 | 14.5 | 1.8 | 3.1 | 1.0 | 0.1 |
|  | 4 | 408 | 100 | 100 | 100 | 91.8 | 93.3 | 99.12 | 52.3 | 57.5 | 83.2 |
|  | 6 | 612 | 100 | 100 | 100 | 81.9 | 87.5 | 84.4 | 41.1 | 47.3 | 66.1 |
|  | 10 | 1020 | 100 | 100 | 99.98 | 69.8 | 77.3 | 77.9 | 27.0 | 32.2 | 54.1 |
| 100 | 2 | 404 | 38.8 | 4.5 | 3.1 | 0.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
|  | 4 | 808 | 100 | 99.73 | 100 | 62.4 | 73.3 | 98.4 | 24.4 | 25.9 | 66.6 |
|  | 6 | 1212 | 100 | 99.94 | 99.91 | 43.4 | 58.2 | 98.2 | 8.0 | 13.4 | 36.8 |
|  | 10 | 2020 | 100 | 99.94 | 100 | 28.1 | 42.5 | 91.7 | 3.0 | 6.2 | 69.1 |
| 200 | 2 | 804 | 65.1 | 8.5 | 2.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
|  | 4 | 1608 | 100 | 99.94 | 99.84 | 5.2 | 11.5 | 83.5 | 0.8 | 1.0 | 25.9 |
|  | 6 | 2412 | 100 | 99.64 | 98.3 | 12.9 | 44.6 | 78.2 | 3.1 | 14.2 | 34.4 |
|  | 10 | 4020 | 100 | 99.86 | 99.54 | 14.6 | 22.2 | 82.0 | 2.2 | 2.2 | 39.9 |

Runs: `emb_c*` and `main_s*` for N = 25, `full_no_length_mix` and `full_no_targeted` for the two extra rows,
`len_n<N>_c<c>` for the others.
