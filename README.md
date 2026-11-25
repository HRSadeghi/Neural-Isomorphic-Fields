# Neural Isomorphic Fields

Code, trained models and results for

> Hamidreza Sadeghi, Saeedeh Momtazi and Reza Safabakhsh.
> **Neural Isomorphic Fields: A Transformer-based Algebraic Numerical Embedding.**
> arXiv:2601.12095, 2026. <https://arxiv.org/abs/2601.12095>

NIF maps a decimal number to a vector and learns, together with the encoder, an addition, a multiplication and a
comparator that work on these vectors directly. The encoder is a Transformer over the digit tokens; its output is read
as one slot per decimal place (52 slots of 10 channels, d = 520, for numbers with up to 25 digits). Both operators
are stacks of Abelian concatenation layers and are commutative for any value of their weights; the comparator is
antisymmetrized. At zero loss the embeddings form an ordered integral domain isomorphic to the decimal fractions,
and its field of fractions is isomorphic to the rationals. For a trained model, the paper bounds the errors of single
operations, chains and algebraic laws by decoding radii, defects and Lipschitz constants.

## Main result

Table 4 of the paper: the tests of Table 3 on the test sets (10,000 tuples per test, fixed seeds), mean ± SD over the
three seeds `main_s0`, `main_s1`, `main_s2`. Accuracy is token accuracy and EM exact match (%), both with the length
chosen by the decoder; the last column gives the exact match when the decoder is told the length of the target.

| Test | n | Accuracy | EM | EM, oracle length |
|:---|---:|---:|---:|---:|
| *Reconstruction* |  |  |  |  |
| &emsp;Reconstruction | 10000 | 100.00 ± 0.00 | 100.00 ± 0.01 | 100.00 ± 0.01 |
| *Addition* |  |  |  |  |
| &emsp;Identity | 10000 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| &emsp;Closure | 10000 | 100.00 ± 0.00 | 100.00 ± 0.01 | 100.00 ± 0.01 |
| &emsp;Inverse | 10000 | 100.00 ± 0.00 | 100.00 ± 0.01 | 100.00 ± 0.01 |
| &emsp;Associativity | 10000 | 100.00 ± 0.00 | 99.98 ± 0.02 | 99.98 ± 0.02 |
| *Multiplication* |  |  |  |  |
| &emsp;Identity | 10000 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| &emsp;Closure | 10000 | 100.00 ± 0.00 | 99.98 ± 0.01 | 99.98 ± 0.01 |
| &emsp;Inverse (units) | 4438 | 92.62 ± 0.16 | 85.22 ± 0.31 | 85.23 ± 0.33 |
| &emsp;Associativity | 10000 | 100.00 ± 0.00 | 99.94 ± 0.02 | 99.94 ± 0.02 |
| *Both operations* |  |  |  |  |
| &emsp;Distributivity | 10000 | 99.99 ± 0.01 | 99.92 ± 0.05 | 99.92 ± 0.05 |
| *Order* |  |  |  |  |
| &emsp;Relation ω̄ | 10000 | – | 99.79 ± 0.10 | 99.79 ± 0.10 |
| &emsp;Antisymmetry of ω | 10000 | – | 99.80 ± 0.13 | 99.80 ± 0.13 |
| &emsp;Transitivity | 19959 | – | 99.99 ± 0.01 | 99.99 ± 0.01 |
| &emsp;Compatibility with + (O1) | 10000 | – | 99.44 ± 0.24 | 99.44 ± 0.24 |
| &emsp;Compatibility with × (O2) | 9759 | – | 100.00 ± 0.00 | 100.00 ± 0.00 |
| *Fraction field* |  |  |  |  |
| &emsp;Addition | 10000 | – | 99.97 ± 0.00 | 99.97 ± 0.00 |
| &emsp;Multiplication | 10000 | – | 99.99 ± 0.01 | 99.99 ± 0.01 |
| &emsp;Order | 10000 | – | 96.54 ± 0.39 | 96.54 ± 0.39 |
| *Chains* |  |  |  |  |
| &emsp;Sums, M = 5 | 500 | – | 99.87 ± 0.23 | 99.87 ± 0.23 |
| &emsp;Sums, M = 10 | 500 | – | 99.80 ± 0.35 | 99.80 ± 0.35 |
| &emsp;Products, M = 5 | 500 | – | 100.00 ± 0.00 | 100.00 ± 0.00 |
| &emsp;Products, M = 10 | 500 | – | 100.00 ± 0.00 | 100.00 ± 0.00 |

Inverse (units) uses all 4438 units of D_25. For the units whose digit products fit inside the slot window the test
reaches 99.77% (see [analysis/](analysis/)).

## Where to find what

| Paper | Experiment | Results and code |
|:---|:---|:---|
| Table 4, Figures 5 and 6 | main model, three seeds | [runs/](runs/), [paper/](paper/) |
| Table 5 | comparison protocol (N = 10, 20, 30) and the AE-only model | [runs/](runs/) |
| Figure 7, Table A.2 | operators trained alone | [runs/](runs/) |
| Figure 8, Table A.3 | ablations of the full model | [runs/](runs/) |
| Figure 9, Table A.4 | number length and embedding dimension | [runs/](runs/), [analysis/](analysis/) |
| Figure 10, Table A.5 | digit code, defects, decoding radii, chains | [analysis/](analysis/) |
| Table A.1 (inference cost) | time and memory per number | [analysis/](analysis/) |
| Table 6 | number vectors read place by place | [reading/](reading/) |
| Table 7, Figures 11 and 12 | language model with NIF operator layers | [downstream/](downstream/) |
| Figure 4 | distributions of the number of digits | [paper/](paper/) |

## Repository layout

```
nif/                 the model, the number sampler and the tests (Python package)
train.py             trains one full NIF model
train_operator.py    trains one operator alone on fixed one-hot digit slots
evaluate.py          runs the tests of the paper on a trained model
run_experiments.py   runs every training of the paper, as listed in configs/experiments.json
configs/             settings of the 43 training runs
runs/                one folder per run: settings, training logs, validation and test results
analysis/            analyses of trained models (lengths, geometry, digit code, inference cost)
results/             outputs of analysis/ and every number of the tables and figures as CSV
paper/               scripts for the tables and figures; their output is in paper/generated/
reading/             the reading experiment (Section 5.3.5, Table 6)
downstream/          the language-model experiment (Section 5.4, Table 7, Figures 11 and 12)
checkpoints/         model weights, downloaded separately (not in git)
```

Every folder has its own README with the commands and the results it holds.

## Installation

Python 3.10 or newer (we used 3.11 and 3.12). All commands below are run from the root of the repository; nothing
has to be installed as a package.

Linux and macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` installs the default PyTorch build for your system. For an NVIDIA GPU, install the CUDA build
first, as described on <https://pytorch.org/get-started/locally/>, for example

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

The language-model experiment needs a few more packages: `pip install -r requirements-lm.txt`.

## Model weights

The weights are not in git (about 2.2 GB for the 43 NIF models and 2.8 GB for each of the two language-model
checkpoints). They can be downloaded from
[Google Drive](https://drive.google.com/drive/folders/1jSvimFyft3ftIpodHPicKbEaBF4jsdvn). Put them into `checkpoints/` as described in
[checkpoints/README.md](checkpoints/README.md). The tables and figures do not need them; the tests, the analyses and
the reading and language-model experiments do.

## Reproducing the paper

**Tables and figures** from the stored logs and results (CPU, no weights needed):

```bash
python paper/make_tables.py        # Tables 4, A.2-A.5, Figure 6, numbers quoted in the text
python paper/make_figures.py       # Figures 4, 5, 7, 8, 9, 10
python reading/make_table.py       # Table 6
python downstream/make_tables.py   # Table 7, Figures 11 and 12 (needs requirements-lm.txt the first time)
```

The output goes to `paper/generated/` and `results/csv/` and is the same as the files in the repository.

**Testing a released model** (a GPU is recommended; `--n 1000` gives a quick check):

```bash
python evaluate.py --ckpt checkpoints/main_s0/best.pt --out my_results/main_s0.json
```

The results of the paper are in `runs/<run>/test_results.json`. The test items are drawn from fixed seeds, so every
model and every repetition is tested on the same numbers. The comparator tests are the most sensitive to the
floating-point arithmetic of the GPU: on another GPU a few borderline pairs can change (a few hundredths of a
percentage point).

**Training.** `python run_experiments.py` runs every training of the paper one after the other, followed by the
tests; a run whose folder `runs/<run>/` already contains `DONE` is skipped. To train a model of the paper again,
use a fresh copy of the repository without the run folder, or give the run another name, for example

```bash
python train.py --name my_main_s0 --set seed=0 steps=12000 patience=4 min_delta=0.2 min_steps=3000
```

Training on a GPU is not bit-for-bit deterministic, so a new run will differ slightly from the released one. One run
of the main configuration takes about 2.2 hours on one NVIDIA L40S.

**Models with N other than 25.** The maximum number of digits N is read from the environment variable
`NIF_N_DIGITS` when `nif.numdata` is imported. `train.py`, `analysis/lengths.py` and `analysis/parts.py` set it
themselves; for the other scripts set it before the call:

```bash
NIF_N_DIGITS=200 python analysis/measure_cost.py --ckpt checkpoints/len_n200_c4/best.pt --digits 200   # Linux, macOS
```

```powershell
$env:NIF_N_DIGITS = "200"; python analysis/measure_cost.py --ckpt checkpoints/len_n200_c4/best.pt --digits 200
```

In `cmd.exe`: `set NIF_N_DIGITS=200` on its own line, then the command.

## Using a trained model

```python
import torch
from evaluate import load

model = load("checkpoints/main_s0/best.pt", "cpu")       # or "cuda"
z = model.encode(["123.45", "-6.7"])                      # tensor of shape (2, 520)
a, b = z[:1], z[1:]
print(model.decode(torch.cat([model.add(a, b), model.mul(a, b)])))   # ['+116.75', '-827.115']
print(model.compare(a, b))                                # probabilities of <, >, =
```

## Citation

```bibtex
@article{sadeghi2026nif,
  title   = {Neural Isomorphic Fields: A Transformer-based Algebraic Numerical Embedding},
  author  = {Sadeghi, Hamidreza and Momtazi, Saeedeh and Safabakhsh, Reza},
  journal = {arXiv preprint arXiv:2601.12095},
  year    = {2026},
  url     = {https://arxiv.org/abs/2601.12095}
}
```
