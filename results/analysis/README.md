# results/analysis

| File | Written by | Contents |
|:---|:---|:---|
| `model_analysis.json` | `analysis/geometry.py` | selected model `main_s0`: slot vectors of every digit at several exponents with a common 2-D projection (Figure 10a), defect, estimated decoding radius and outcome of 800 closure pairs per operation (Figure 10c) |
| `extra.json` | `analysis/extra.py`, `analysis/digit_code.py` | per run: inverse test split by the slot window, kinds of reconstruction errors, digit code (variance by harmonic, effective dimension, agreement between exponents), reconstruction with the true length |
| `cost.jsonl` | `analysis/measure_cost.py` | time per number and peak GPU memory of each part of `main_s0` (Table A.1) |

The numbers that the paper quotes from these files are listed in [analysis/README.md](../../analysis/README.md).
