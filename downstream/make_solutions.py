"""The base model's own step-by-step solutions of training problems (the word-problem training data of the operator
layers): GSM8K training problems (--source gsm8k) or the word problems from various domains (--source domains).

    python downstream/make_solutions.py --source gsm8k
    python downstream/make_solutions.py --source domains --passes 0,0.7 --keep 2

For every training problem the untrained language model writes a solution, with the prompt that the trained models
get (the question with canonical numbers and the step-by-step instruction). A pass is greedy (temperature 0) or
sampled (temperature > 0). A solution is kept if it contains '####' and its final number equals the reference answer
(or, for the domain problems, rounds to it at the answer's decimal places; the kept target ends with the answer
itself); at most --keep different solutions per problem (the greedy one first). Output: <data-dir>/gsm8k_solutions.json.gz
or domain_solutions.json.gz (the examples, numbers as Num parts, and statistics) and a .log next to it. Every generated
solution is appended to a .partial.jsonl next to it; a restarted run continues from it.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402

import data as D  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="gsm8k", choices=["gsm8k", "domains"])
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--passes", default="0,0.7,0.7", help="temperatures of the passes (0: greedy)")
    ap.add_argument("--keep", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="only the first n problems (tests)")
    a = ap.parse_args()
    stem = "gsm8k_solutions" if a.source == "gsm8k" else "domain_solutions"
    out_json = os.path.join(a.data_dir, stem + ".json.gz")
    part_path = os.path.join(a.data_dir, stem + ".partial.jsonl")
    log_path = os.path.join(a.data_dir, stem + ".log")

    def log(msg):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    os.environ.setdefault("NIF_DS_KERNELS", "fla")
    from llm import NumLLM
    if a.source == "gsm8k":
        train = D.load_gsm8k_cot(a.data_dir, log)["train"]
        if a.limit:
            train = train[:a.limit]
        exs = [D.cot_parts(dict(r, task="gsm")) for r in train]
    else:                                              # word problems from various domains
        import domain_problems
        train = domain_problems.load(a.data_dir)["train"]
        if a.limit:
            train = train[:a.limit]
        exs = [domain_problems.example(r) for r in train]
    done = {}
    if os.path.exists(part_path):
        for line in open(part_path, encoding="utf-8"):
            try:
                x = json.loads(line)
                done[(x["i"], x["pass"])] = x
            except ValueError:
                pass
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    llm = NumLLM(a.model, dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev)
    llm.eval()
    log(f"{len(exs)} problems, passes {a.passes}, {len(done)} solutions already generated; {llm.check_cache(exs[:4], 6)}")
    temps = [float(t) for t in a.passes.split(",")]
    order = sorted(range(len(exs)), key=lambda i: len(D.render(exs[i]["user"])[0]))
    for p, temp in enumerate(temps):
        llm.temperature = temp
        llm.sample_gen = torch.Generator(device=dev).manual_seed(1000 + p)
        todo = [i for i in order if (i, p) not in done]
        t0, n_ok = time.time(), 0
        for s0 in range(0, len(todo), a.batch):
            idx = todo[s0:s0 + a.batch]

            def gen(ids):
                try:
                    return [t for t, _ in llm.generate([exs[i] for i in ids], a.max_new)]
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    h = len(ids) // 2
                    return gen(ids[:h]) + gen(ids[h:])
            texts = gen(idx)
            with open(part_path, "a", encoding="utf-8") as f:
                for i, t in zip(idx, texts):
                    ok = "####" in t and bool(D.score(exs[i], t))
                    n_ok += ok
                    x = {"i": i, "pass": p, "temp": temp, "ok": ok, "text": t}
                    done[(i, p)] = x
                    f.write(json.dumps(x) + "\n")
            log(f"pass {p} (temperature {temp}): {min(s0 + a.batch, len(todo))}/{len(todo)} problems, "
                f"correct so far {100 * n_ok / max(1, min(s0 + a.batch, len(todo))):.1f}%, {time.time() - t0:.0f} s")

    examples, stats = [], {"problems": len(exs), "passes": temps}
    per_pass = [sum(done.get((i, p), {}).get("ok", False) for i in range(len(exs))) for p in range(len(temps))]
    stats["correct_per_pass"] = [round(100 * c / len(exs), 2) for c in per_pass]
    solved = 0
    for i, r in enumerate(train):
        texts = []
        for p in range(len(temps)):
            x = done.get((i, p))
            if x and x["ok"]:
                body = x["text"].split("####")[0].strip()
                if body and body not in texts:
                    texts.append(body)
        solved += bool(texts)
        for body in texts[:a.keep]:
            e = dict(task=exs[i]["task"] if a.source == "domains" else "gsm_cot", size=exs[i]["size"] if a.source ==
                     "domains" else 0, user=exs[i]["user"], asst=D.split_numbers(body) + ["\n#### ", D.Num(r["answer"])],
                     answer=r["answer"], question=r["question"], trainable=True, source="self")
            if a.source == "domains":
                e.update(decimals=r["decimals"], id=r["id"], difficulty=r["difficulty"])
            examples.append(e)
    stats.update(solved=solved, solved_pct=round(100 * solved / max(1, len(exs)), 2), examples=len(examples))
    if a.source == "domains":                          # what the model solves, by difficulty (selection of the data)
        import domain_problems
        for d in domain_problems.DIFF:
            idx = [i for i, r in enumerate(train) if r["difficulty"] == d]
            ok = [any(done.get((i, p), {}).get("ok", False) for p in range(len(temps))) for i in idx]
            stats[f"solved_{d}"] = round(100 * sum(ok) / max(1, len(ok)), 2)
            stats[f"greedy_{d}"] = round(100 * sum(done.get((i, 0), {}).get("ok", False) for i in idx) / max(1, len(idx)), 2)
        stats["problems_with_solution"] = len({e["id"] for e in examples})
    raw = {"train": [dict(e, user=D.pack(e["user"]), asst=D.pack(e["asst"])) for e in examples], "stats": stats}
    with gzip.open(out_json + ".tmp", "wt", encoding="utf-8") as f:
        json.dump(raw, f)
    os.replace(out_json + ".tmp", out_json)
    log(f"written {out_json}: {json.dumps(stats)}")


if __name__ == "__main__":
    main()
