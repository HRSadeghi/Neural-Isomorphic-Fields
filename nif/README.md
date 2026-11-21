# nif

The model and everything it needs. The modules are imported one by one (`from nif.model import NIF`); the package
itself imports nothing, because `numdata` fixes the maximum number of digits N when it is first imported (25, or the
value of `NIF_N_DIGITS`).

| Module | Contents |
|:---|:---|
| `numdata.py` | numbers of D_N as (sign, digits) tensors: the sampler of Section 5.1, admissible pairs, exact sums and products (training targets only), the hint targets of the staged product, token sequences |
| `layers.py` | RMSNorm, rotary embeddings, attention, SwiGLU with periodic units, the Abelian concatenation (AC) layer, the Abelian neural operator, the staged multiplication operator and its hint heads, the perceptron AC layers of the ablations |
| `model.py` | `NIF`: encoder φ, operators + and ×, decoder φ† with its length head, comparator ω |
| `train_utils.py` | helpers of the training loop (exact order of digit arrays, batched encoding, decoder loss, exact match) |
| `convergence.py` | the stopping rule (validation every 500 steps, patience, minimum improvement) |
| `algebra_tests.py` | the tests of Table 3 and the quantities of Section 3.6 (defects, decoding radii, Lipschitz constants, chains) |
| `numbers.py` | exact number handling for the tests with `fractions.Fraction`, canonical strings, units of D_N, fraction pairs |
| `backend.py` | small helpers that work on torch tensors and NumPy arrays alike |
| `stats.py` | exact McNemar test and Holm–Bonferroni correction |
| `paths.py` | folders of the repository; `checkpoint(run)` returns `runs/<run>/best.pt` after a local training, else `checkpoints/<run>/best.pt` |

A model with N = 25 and c = 10 has 18.8M parameters: encoder 5.01M, decoder 6.08M, addition 2.12M, multiplication
4.23M, comparator 1.34M and the hint heads 0.02M.

`python -m nif.numbers` runs a short self-test of the exact number code.
