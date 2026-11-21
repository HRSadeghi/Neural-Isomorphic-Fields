# downstream/data

| File | Contents |
|:---|:---|
| `gsm8k_solutions.json.gz` | 7,508 correct step-by-step solutions written by the base model for 4,815 of the 7,473 GSM8K training problems |
| `domain_solutions.json.gz` | 16,779 correct solutions written by the base model for word problems from various domains |
| `domain_problems.json.gz` | the 43,263 word problems from various domains (question, answer, decimal places of the answer, difficulty) |

The solutions come from one greedy pass and one sampled pass (temperature 0.7) of the untrained base model, with the
prompt that the trained models get; a solution is kept if its final number equals the reference answer, at most two
different solutions per problem. They are rebuilt with

```bash
python downstream/make_solutions.py --source gsm8k --passes 0,0.7 --keep 2
python downstream/make_solutions.py --source domains --passes 0,0.7 --keep 2
```

GSM8K (`openai/gsm8k`) and GSM-Ranges (`guactastesgood/GSM-Ranges`) are downloaded from Hugging Face the first time
they are needed and cached here as `gsm8k_processed.json`, `gsm8k_cot_processed.json` and `gsm_ranges_processed.json`
(not in git).
