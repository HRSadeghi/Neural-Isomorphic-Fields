# reading

The reading experiment of Section 5.3.5 (Table 6): can a Transformer that adds and subtracts integers use number
vectors to handle numbers longer than in training? A small Transformer (rotary positions, trained from scratch on the
CPU) adds and subtracts non-negative integers written as digit tokens and writes the answer from the units digit; the
training operands have at most 12 digits, the test goes up to 20. The vectors are read place by place: the position
that writes the digit of place p receives slots p and p−1 of both operand vectors through maps shared by all places.

| Condition | Input |
|:---|:---|
| `text_lsb` | digit tokens only |
| `read_onehot` | + the one-hot digit of every place |
| `read_f64fone` | + FoNE features computed from the number as a float64 |
| `read_fone` | + FoNE features computed with exact digit arithmetic |
| `read_nif` | + the frozen NIF encoder of `main_s0` |

```bash
python reading/run.py --list          # commands and what is finished
python reading/run.py                 # the five runs of the table (finished ones are skipped)
python reading/run.py --jobs 3        # three runs at the same time
python reading/make_table.py          # paper/generated/tab_reader.tex, reader_numbers.tex, results/csv/reader.csv
```

Each run uses 2 CPU threads, 30,000 training problems and 4000 steps, and takes one to two hours. `read_nif` needs
`checkpoints/main_s0/best.pt`; the vectors of each condition are cached in `reading/cache/`. The logs of the runs of
the paper are in `logs/` (`<condition>_s0.log`, one JSON line per 1000 steps); `make_table.py` reads their last line.
A single run: `python reading/experiment.py <condition> <steps> <seed> [threads] [training problems] [vocabulary size]`.

## Results (Table 6)

Exact match (%) by the number of digits of the longer operand; 200 test problems per length, one run.

| Input of the model | 4 | 8 | 12 | 14 | 16 | 20 |
|:---|---:|---:|---:|---:|---:|---:|
| Digit tokens only | 98.5 | 99.0 | 92.5 | 0.0 | 0.0 | 0.0 |
| + one-hot digits, read place by place | 99.0 | 99.0 | 99.0 | 100.0 | 98.5 | 89.0 |
| + FoNE (float64), read place by place | 99.5 | 99.0 | 98.5 | 99.0 | 95.0 | 0.0 |
| + FoNE (exact), read place by place | 98.5 | 99.5 | 98.0 | 98.0 | 95.0 | 88.5 |
| + NIF (ours), read place by place | 99.0 | 99.5 | 99.5 | 99.5 | 98.0 | 92.5 |

Lengths 4, 8, 12 are within the training lengths, 14, 16, 20 are longer.
