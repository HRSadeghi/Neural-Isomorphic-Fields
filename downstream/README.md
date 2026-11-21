# downstream

The language-model experiment of Section 5.4 (Table 7, Figures 11 and 12). The base model is Qwen3.5-0.8B
(752M parameters). After the question, a workspace lists every number of the prompt as a value token between two tag
tokens; the value token is the NIF vector of the number (`main_s0`, frozen) through a trained adapter. After decoder
layers 9 and 15 an operator layer selects registers by pointers, the frozen NIF operators compute sums, differences
and products, and the results are written back into the hidden states. The answer is decoded from a mixture of the
language model and the digits of the registers given by the NIF decoder. The operator layers add 10.7M parameters.

| File | Contents |
|:---|:---|
| `data.py` | tasks: synthetic arithmetic, templated word problems of controlled number length, GSM8K, GSM-Ranges, the training mixture, the test cells |
| `domain_problems.py` | the word problems from various domains of the training mixture |
| `make_solutions.py` | the base model's own step-by-step solutions of the training problems |
| `llm.py` | the language model with numbers written as digits |
| `oplayers.py` | the NIF operator layers |
| `train.py` | training and tests of one run |
| `run.py` | the runs of the paper |
| `make_tables.py` | Table 7, Figures 11 and 12, `llm_numbers.tex`, `results/csv/llm*.csv` |
| `data/` | training data (see [data/README.md](data/README.md)) |
| `runs/` | outputs of the runs of the paper (see [runs/README.md](runs/README.md)) |

## Running

Install `requirements-lm.txt` in addition to `requirements.txt`. A run needs one GPU with about 70 GB of memory
(smaller GPUs work, the batches are then split automatically, but slower); on Linux `flash-linear-attention` speeds up
the linear-attention layers of the base model. GSM8K and GSM-Ranges are downloaded from Hugging Face on first use and
cached in `data/`.

```bash
python downstream/run.py                          # list the runs and what is finished
python downstream/run.py operator_layers_test     # final test and interventions of the released model
python downstream/run.py base_model operator_layers operator_layers_10000 --out my_runs
python downstream/make_tables.py                  # tables and figures from downstream/runs/
```

| Run | What it does |
|:---|:---|
| `base_model` | Qwen3.5-0.8B as released, all tests |
| `operator_layers` | 6000 training steps with the operator layers; the complete test every 500 steps, then the final test and the interventions |
| `operator_layers_10000` | `operator_layers` continued to 10,000 steps from its checkpoint, with a second learning-rate cycle (Figure 11c) |
| `operator_layers_test` | the final test and the interventions of `checkpoints/language_model/operator_layers_6000.pt`, without training |

Training: all weights of the base model are trained together with the adapters for 6000 steps (AdamW, batch size 32,
learning rate 1e-5 for the base model and 1e-3 for the adapters, 100 warm-up steps, cosine decay); NIF stays frozen.
It takes about 3.2 hours on a 71 GB partition of an NVIDIA H200 GPU. The training mixture is 40% arithmetic problems,
15% templated word problems with step-by-step solutions, 15% the base model's own correct solutions of
4,815 of the 7,473 GSM8K training problems (7,508 solutions) and 30% its own
solutions of word problems from various domains.

Test sets: GSM-Ranges (100 GSM8K test problems at six levels of number size, two variants of each problem per level)
and 8 held-out templates of addition and subtraction word problems with numbers of 1 to 20 digits (100 problems for
each of the lengths 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 20). Both models solve each problem step by step and decode
greedily (at most 512 tokens); an answer is correct if its final number equals the reference value.

## Results (Table 7)

Exact match (%) of the final answer, base model against the model trained with NIF on the same problems. Operators
off: the model trained with NIF, with the operator layers and register outputs switched off. p: exact McNemar test of
the base model and the model trained with NIF, Holm–Bonferroni adjusted over the rows; bold: p < 0.05.

| Test set | Problems | Base model | Trained with NIF | Operators off | p |
|:---|---:|---:|---:|---:|---:|
| GSM-Ranges, level 1 | 200 | 59.5 | 55.0 | 55.5 | 0.92 |
| GSM-Ranges, level 2 | 200 | 53.5 | 53.5 | 53.0 | 1.00 |
| GSM-Ranges, level 3 | 200 | 43.0 | 46.5 | 46.0 | 0.92 |
| GSM-Ranges, level 4 | 200 | 39.0 | 44.5 | 45.0 | 0.77 |
| GSM-Ranges, level 5 | 200 | 29.5 | 43.0 | 42.5 | **0.004** |
| GSM-Ranges, level 6 | 200 | 28.0 | 43.0 | 40.5 | **<0.001** |
| Templated problems, numbers of 1–12 digits | 800 | 54.2 | 85.1 | 85.2 | **<0.001** |
| Templated problems, numbers of 14–20 digits | 300 | 1.3 | 39.7 | 12.0 | **<0.001** |

Training for 10,000 instead of 6000 steps (`operator_layers_10000`) changes no row of the table significantly
(smallest adjusted p = 0.06).

## Interventions (Figure 12)

The trained model (step 6000) as trained, with the operator layers, NIF vectors and register outputs switched off (the
trained language model writes the answer alone), and with the vectors of other numbers of the same problem in the
workspace. Exact match (%); p: McNemar test of "as trained" against "operator layers off", Holm adjusted.

| Test set | Problems | As trained | Operator layers off | Vectors of other numbers | p (on vs off, Holm) |
|:---|---:|---:|---:|---:|---:|
| Addition, 1–12 digits | 800 | 100.0 | 98.0 | 13.2 | **<0.001** |
| Addition, 14–20 digits | 300 | 98.7 | 18.0 | 2.3 | **<0.001** |
| Subtraction, 1–12 digits | 800 | 100.0 | 98.6 | 4.8 | **0.004** |
| Subtraction, 14–20 digits | 300 | 97.7 | 15.0 | 0.3 | **<0.001** |
| Multiplication, factors of 1–6 digits | 600 | 100.0 | 68.2 | 13.3 | **<0.001** |
| Multiplication, factors of 7–8 digits | 200 | 100.0 | 43.5 | 4.5 | **<0.001** |
| Templated problems, numbers of 1–12 digits | 800 | 85.1 | 85.2 | 58.5 | 1.00 |
| Templated problems, numbers of 14–20 digits | 300 | 39.7 | 12.0 | 8.7 | **<0.001** |
| GSM-Ranges, levels 1–3 | 600 | 51.7 | 51.5 | 51.3 | 1.00 |
| GSM-Ranges, levels 4–6 | 600 | 43.5 | 42.7 | 41.7 | 1.00 |
