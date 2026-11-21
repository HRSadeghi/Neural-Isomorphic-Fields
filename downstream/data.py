"""Tasks of the language-model experiment: synthetic arithmetic, templated word problems of controlled number length,
GSM8K and GSM-Ranges (GSM8K problems with numbers of up to seven digits), and the training mixture.

An example is a dict
    task, size, user (list of parts), asst (list of parts), answer (canonical string or "A"/"B")
where a part is either a string or a number Num("12.5"); numbers are written as digits.
Only the Python standard library is needed here (GSM8K additionally needs `datasets`, GSM-Ranges `huggingface_hub`).
"""
from __future__ import annotations

import gzip
import json
import os
import random
import re
from decimal import Decimal, InvalidOperation, getcontext

getcontext().prec = 120                      # exact for every sum / product used here

MAX_INT, MAX_FRAC = 25, 26                   # the window of the NIF embedding (N = 25)


class Num(str):
    """A number inside a prompt or an answer (canonical decimal string)."""


# ------------------------------------------------------------------------------------- number strings
def canon(s):
    """Canonical decimal string ('-12.5', '0', '300'), or None if s is not a finite decimal in the window."""
    if s is None:
        return None
    s = str(s).strip().replace(",", "")
    if not re.fullmatch(r"[+-]?(\d+\.?\d*|\.\d+)", s):
        return None
    neg = s.startswith("-")
    s = s.lstrip("+-")
    ip, fp = (s.split(".") + [""])[:2]
    ip, fp = ip.lstrip("0") or "0", fp.rstrip("0")
    if len(ip) > MAX_INT or len(fp) > MAX_FRAC:
        return None
    out = ip + ("." + fp if fp else "")
    return "-" + out if neg and out != "0" else out


def dec(s):
    return Decimal(s)


def dstr(d):
    return canon(format(d, "f"))


def n_parts(s):
    """(integer digits, fractional digits) of a canonical string."""
    s = s.lstrip("-")
    ip, fp = (s.split(".") + [""])[:2]
    return len(ip), len(fp)


def last_number(text):
    """The last decimal number written in a text (thousands separators removed), canonical, or None."""
    m = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text or "")
    return canon(m[-1]) if m else None


def rnd_number(rng, n_int, n_frac=0, neg=False):
    ip = str(rng.randint(1, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(n_int - 1))
    fp = "".join(str(rng.randint(0, 9)) for _ in range(n_frac - 1)) + str(rng.randint(1, 9)) if n_frac else ""
    return Num(canon(("-" if neg else "") + ip + ("." + fp if fp else "")))


def operand(rng, size, p_dec=0.3, exact=False):
    """A non-negative number with at most `size` integer digits (exactly `size` if exact); with
    probability p_dec it has 1..min(4, size) fractional digits."""
    n_int = size if exact else rng.randint(1, size)
    n_frac = rng.randint(1, min(4, size)) if rng.random() < p_dec else 0
    return rnd_number(rng, n_int, n_frac)


# ------------------------------------------------------------------------------------- synthetic tasks
ONLY = "\nAnswer with the number only."


def ex_add(rng, size):
    a, b = operand(rng, size, exact=True), operand(rng, size)
    if rng.random() < 0.5:
        a, b = b, a
    c = Num(dstr(dec(a) + dec(b)))
    return dict(task="add", size=size, user=["Compute: ", a, " + ", b, ONLY], asst=[c], answer=str(c))


def ex_sub(rng, size):
    a, b = operand(rng, size, exact=True), operand(rng, size)
    if rng.random() < 0.5:
        a, b = b, a
    c = Num(dstr(dec(a) - dec(b)))
    return dict(task="sub", size=size, user=["Compute: ", a, " - ", b, ONLY], asst=[c], answer=str(c))


def ex_mul(rng, size):
    """Integers: `size` digits times 1..min(size, 3) digits (one forward pass, no scratchpad)."""
    a = rnd_number(rng, size)
    b = rnd_number(rng, rng.randint(1, min(size, 3)))
    if rng.random() < 0.5:
        a, b = b, a
    c = Num(dstr(dec(a) * dec(b)))
    return dict(task="mul", size=size, user=["Compute: ", a, " * ", b, ONLY], asst=[c], answer=str(c))


def ex_cmp(rng, size):
    a = operand(rng, size, exact=True)
    if rng.random() < 0.5:                                   # near pair: one digit changed
        digits = [i for i, ch in enumerate(a) if ch.isdigit()]
        while True:
            i = rng.choice(digits)
            ch = str(rng.randint(0, 9))
            b = canon(a[:i] + ch + a[i + 1:])
            if b is not None and b != a:
                break
        b = Num(b)
    else:
        b = operand(rng, size)
        while b == a:
            b = operand(rng, size)
    if rng.random() < 0.5:
        a, b = b, a
    ans = "A" if dec(a) > dec(b) else "B"
    return dict(task="cmp", size=size, user=["Which number is larger?\nA: ", a, "\nB: ", b, "\nAnswer with A or B."],
                asst=[ans], answer=ans)


def ex_chain(rng, size, steps=None):
    """a1 +- a2 +- ... : the answer lists every running result (the latent scratchpad)."""
    steps = steps or rng.randint(2, 4)
    terms = [operand(rng, size, p_dec=0.0, exact=(i == 0)) for i in range(steps + 1)]
    ops = [rng.choice("+-") for _ in range(steps)]
    user, asst, cur = ["Compute from left to right and write every running result, separated by ' ; ': ", terms[0]], [], dec(terms[0])
    for op, t in zip(ops, terms[1:]):
        user += [f" {op} ", t]
        cur = cur + dec(t) if op == "+" else cur - dec(t)
        asst += ([" ; "] if asst else []) + [Num(dstr(cur))]
    return dict(task="chain", size=steps, user=user, asst=asst, answer=str(asst[-1]))


# ------------------------------------------------------------------------------------- word problems of controlled length
# GSM8K has almost no long numbers (96% of its numbers have at most three digits), so the effect of the number
# length inside a word problem is measured with these templates: additions and subtractions only (the operations
# that are trained up to max_size digits), one to three steps, and the digits of the quantities as the size.
# The stories of the test split are never seen in training; every structure occurs in both splits.
WP_HINT = "\nWrite the intermediate results and the final answer, separated by ' ; '."
S_ADD, S_SUB = [("a", "+", "b")], [("a", "-", "b")]
S_ADD3 = [("a", "+", "b"), ("r1", "+", "c")]
S_SUB3 = [("a", "-", "b"), ("r1", "-", "c")]
S_ADDSUB = [("a", "+", "b"), ("r1", "-", "c")]
S_SUBADD = [("a", "-", "b"), ("r1", "+", "c")]
S_REST = [("b", "+", "c"), ("a", "-", "r1")]
S_NET = [("a", "+", "b"), ("c", "+", "d"), ("r1", "-", "r2")]
WP = {  # split: [(text, steps, decimals allowed)]
    "train": [
        ("A national archive holds {a} scanned pages. This year it scanned {b} more pages. How many scanned pages does the archive hold now?", S_ADD, False),
        ("A reservoir contained {a} liters of water. During the rainy season {b} liters were added. How many liters does the reservoir contain now?", S_ADD, True),
        ("A warehouse had {a} screws in stock. It shipped {b} screws to its customers. How many screws are left in the warehouse?", S_SUB, False),
        ("A probe must travel {a} kilometers to reach its orbit. It has already travelled {b} kilometers. How many kilometers remain?", S_SUB, True),
        ("A charity collected {a} dollars in January, {b} dollars in February and {c} dollars in March. How many dollars did it collect in the three months?", S_ADD3, True),
        ("Three farms harvested {a}, {b} and {c} grains of rice. How many grains of rice did they harvest together?", S_ADD3, False),
        ("A data center has {a} bytes of free storage. A backup uses {b} bytes and a video archive uses {c} bytes. How many bytes stay free?", S_SUB3, False),
        ("A printing house started the week with {a} sheets of paper. It used {b} sheets on Monday and {c} sheets on Tuesday. How many sheets are left?", S_SUB3, False),
        ("A bank account held {a} dollars. A deposit of {b} dollars was made and then {c} dollars were withdrawn. How many dollars are in the account now?", S_ADDSUB, True),
        ("A stadium sold {a} tickets online and {b} tickets at the gate. Later {c} tickets were returned. How many tickets remain sold?", S_ADDSUB, False),
        ("A city had {a} trees. A storm destroyed {b} trees, and afterwards volunteers planted {c} new trees. How many trees does the city have now?", S_SUBADD, False),
        ("A tank held {a} liters of fuel. A truck took {b} liters, and then a delivery added {c} liters. How many liters are in the tank now?", S_SUBADD, True),
        ("A factory must produce {a} bottles this year. It produced {b} bottles in the first half of the year and {c} bottles in the third quarter. How many bottles must it still produce?", S_REST, False),
        ("A road of {a} meters is being paved. One team paved {b} meters and another team paved {c} meters. How many meters are not paved yet?", S_REST, True),
        ("A company earned {a} dollars from sales and {b} dollars from services. It paid {c} dollars in salaries and {d} dollars in rent. How many dollars of profit did the company make?", S_NET, True),
        ("Two rivers bring {a} and {b} liters of water into a lake each day. Evaporation removes {c} liters and a canal removes {d} liters each day. By how many liters does the lake grow each day?", S_NET, False),
    ],
    "test": [
        ("An observatory catalogued {a} stars last year and {b} new stars this year. How many stars are in its catalogue in total?", S_ADD, False),
        ("A mine holds an estimated {a} grams of copper. So far {b} grams have been extracted. How many grams of copper are still in the mine?", S_SUB, True),
        ("A postal service delivered {a} letters in spring, {b} letters in summer and {c} letters in autumn. How many letters did it deliver in these three seasons?", S_ADD3, False),
        ("An airline had {a} unsold seats for the year. It sold {b} seats in a spring sale and {c} seats in a summer sale. How many seats are still unsold?", S_SUB3, False),
        ("A museum counted {a} visitors in its first decade and {b} visitors in its second decade. An audit then removed {c} duplicate entries. How many visitors remain in the count?", S_ADDSUB, False),
        ("A forest stored {a} kilograms of carbon. A fire released {b} kilograms, and new growth later stored {c} kilograms more. How many kilograms of carbon does the forest store now?", S_SUBADD, True),
        ("A telescope has to take {a} images for a sky survey. It took {b} images in the first year and {c} images in the second year. How many images does it still have to take?", S_REST, False),
        ("A port loaded {a} containers onto ships and {b} containers onto trains. It received {c} containers by sea and {d} containers by road. How many more containers did it load than it received?", S_NET, False),
    ],
}


def ex_wp(rng, size, split="train"):
    """A templated word problem whose first quantity has exactly `size` integer digits (the others at most
    `size`); the target lists the result of every step, like the GSM8K targets. No result is negative."""
    text, steps, decimals = rng.choice(WP[split])
    names = re.findall(r"\{(\w)\}", text)
    for _ in range(10000):
        vals = {n: operand(rng, size, p_dec=0.3 if decimals else 0.0, exact=(n == "a")) for n in names}
        env, res = {n: dec(v) for n, v in vals.items()}, []
        for i, (x, op, y) in enumerate(steps, 1):
            env[f"r{i}"] = env[x] + env[y] if op == "+" else env[x] - env[y]
            res.append(env[f"r{i}"])
        if min(res) >= 0:
            break
    else:
        raise RuntimeError("no non-negative word problem found")
    user = []
    for piece in re.split(r"(\{\w\})", text):
        user.append(vals[piece[1]] if re.fullmatch(r"\{\w\}", piece) else piece)
    asst = []
    for r in res:
        asst += ([" ; "] if asst else []) + [Num(dstr(r))]
    question = "".join(str(p) for p in user)
    return dict(task="wp", size=size, user=user + [WP_HINT], asst=asst, answer=str(asst[-1]), question=question)


MAKERS = {"add": ex_add, "sub": ex_sub, "mul": ex_mul, "cmp": ex_cmp, "chain": ex_chain, "wp": ex_wp}
MAX_SIZE = {"mul": 6, "chain": 6}                      # largest training size of a task (default: cfg max_size)
CHAIN_TRAIN_STEPS = 4                                  # chains are trained with 2..4 steps


def make(task, rng, size):
    return MAKERS[task](rng, size)


# ------------------------------------------------------------------------------------- GSM8K
NUM_RE = re.compile(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?(?![\w])")
GSM_HINT = "\nWrite the intermediate results and the final answer, separated by ' ; '."
GSM_COT = "\nSolve the problem step by step and finish with '#### ' followed by the final number."


def split_numbers(text):
    """Text -> parts, every number (without thousands separators) as a Num."""
    parts, pos = [], 0
    for m in NUM_RE.finditer(text):
        v = canon(m.group(0))
        if v is None:
            continue
        if m.start() > pos:
            parts.append(text[pos:m.start()])
        parts.append(Num(v))
        pos = m.end()
    if pos < len(text):
        parts.append(text[pos:])
    return parts


def gsm_example(question, solution, max_steps=8):
    """One GSM8K problem. The target is the list of the results of the calculator annotations <<a*b=c>>
    followed by the final answer (not repeated if it is the last result). Returns None if a value is not a
    finite decimal in the window; `trainable` is False if there are more than max_steps values."""
    final = canon(solution.split("####")[-1])
    if final is None:
        return None
    res = [canon(x) for x in re.findall(r"<<[^<>]*=([^<>]*)>>", solution)]
    if any(r is None for r in res):
        res = []                                       # e.g. a fraction as a result: keep the final answer only
    if not res or dec(res[-1]) != dec(final):
        res.append(final)
    asst = []
    for r in res:
        asst += ([" ; "] if asst else []) + [Num(r)]
    return dict(task="gsm", size=len(res), user=split_numbers(question.strip()) + [GSM_HINT], asst=asst, answer=final,
                trainable=len(res) <= max_steps, question=question.strip())


def load_gsm8k(data_dir, log=print):
    """Download GSM8K (Hugging Face `openai/gsm8k`, configuration `main`) once and cache the processed
    examples as JSON in data_dir."""
    path = os.path.join(data_dir, "gsm8k_processed.json")
    if os.path.exists(path):
        raw = json.load(open(path, encoding="utf-8"))
    else:
        from datasets import load_dataset
        os.makedirs(data_dir, exist_ok=True)
        raw, stats = {}, {}
        for split in ("train", "test"):
            ds = load_dataset("openai/gsm8k", "main", split=split)
            rows = [gsm_example(r["question"], r["answer"]) for r in ds]
            raw[split] = [r for r in rows if r is not None]
            stats[split] = {"problems": len(rows), "kept": len(raw[split]),
                            "trainable": sum(r["trainable"] for r in raw[split])}
            for r in raw[split]:                       # numbers are stored as ["n", value]
                r["user"], r["asst"] = pack(r["user"]), pack(r["asst"])
        raw["stats"] = stats
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(raw, f)
        os.replace(path + ".tmp", path)
        log(f"GSM8K downloaded and processed: {json.dumps(stats)}")
    out = {split: [dict(r, user=unpack(r["user"]), asst=unpack(r["asst"])) for r in raw[split]] for split in ("train", "test")}
    out["stats"] = raw["stats"]
    return out


def pack(parts):
    return [["n", str(p)] if isinstance(p, Num) else p for p in parts]


def unpack(parts):
    return [Num(p[1]) if isinstance(p, list) else p for p in parts]


def cot_example(r):
    """Zero-shot variant of a word problem (free chain of thought, answer after ####); task '<task>_cot'."""
    return dict(task=r["task"] + "_cot", size=r["size"], user=[r["question"] + GSM_COT], asst=["#### " + r["answer"]],
                answer=r["answer"])


# ------------------------------------------------------------------------------------- step-by-step solutions
# The model writes its reasoning in words and numbers and ends with '#### <answer>'. The numbers of the prompt and
# of the solution are Num parts, so that the operator layers see the numbers of the prompt (workspace) and the numbers
# that the model writes (operations on generated numbers, oplayers.py).
def cot_parts(r):
    """A word problem (GSM8K, GSM-Ranges, templated) with its numbers as Num parts and the step-by-step instruction;
    the target is only the final answer (for tests)."""
    return dict(task=base_task(r["task"]) + "_cot", size=r["size"], user=split_numbers(r["question"]) + [GSM_COT],
                asst=["#### ", Num(r["answer"])], answer=r["answer"], question=r["question"])


def gsm_cot_example(question, solution):
    """A GSM8K problem with its reference solution as the target: the calculator annotations <<...>> are removed,
    the numbers are Num parts (thousands separators removed), the last line is '#### <answer>'."""
    final = canon(solution.split("####")[-1])
    if final is None:
        return None
    body = re.sub(r"<<[^<>]*>>", "", solution.split("####")[0]).strip()
    return dict(task="gsm_cot", size=0, user=split_numbers(question.strip()) + [GSM_COT],
                asst=split_numbers(body) + ["\n#### ", Num(final)], answer=final, question=question.strip(), trainable=True)


def load_gsm8k_cot(data_dir, log=print):
    """GSM8K with step-by-step targets (train and test), downloaded once and cached as JSON in data_dir."""
    path = os.path.join(data_dir, "gsm8k_cot_processed.json")
    if os.path.exists(path):
        raw = json.load(open(path, encoding="utf-8"))
    else:
        from datasets import load_dataset
        os.makedirs(data_dir, exist_ok=True)
        raw, stats = {}, {}
        for split in ("train", "test"):
            ds = load_dataset("openai/gsm8k", "main", split=split)
            rows = [gsm_cot_example(r["question"], r["answer"]) for r in ds]
            raw[split] = [r for r in rows if r is not None]
            stats[split] = {"problems": len(rows), "kept": len(raw[split]),
                            "numbers_per_solution": round(sum(sum(isinstance(p, Num) for p in r["asst"]) for r in raw[split])
                                                          / max(1, len(raw[split])), 2)}
            for r in raw[split]:
                r["user"], r["asst"] = pack(r["user"]), pack(r["asst"])
        raw["stats"] = stats
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(raw, f)
        os.replace(path + ".tmp", path)
        log(f"GSM8K with step-by-step solutions downloaded and processed: {json.dumps(stats)}")
    out = {split: [dict(r, user=unpack(r["user"]), asst=unpack(r["asst"])) for r in raw[split]] for split in ("train", "test")}
    out["stats"] = raw["stats"]
    return out


def load_self_cot(path):
    """The model's own correct step-by-step solutions of training problems (make_solutions.py); .json or .json.gz."""
    raw = json.load((gzip.open if path.endswith(".gz") else open)(path, "rt", encoding="utf-8"))
    return [dict(r, user=unpack(r["user"]), asst=unpack(r["asst"])) for r in raw["train"]], raw.get("stats", {})


def ex_wp_cot(rng, size, split="train"):
    """A templated word problem (as ex_wp) whose target writes every step as '<x> <op> <y> = <result>' and ends with
    '#### <answer>'."""
    text, steps, decimals = rng.choice(WP[split])
    names = re.findall(r"\{(\w)\}", text)
    for _ in range(10000):
        vals = {n: operand(rng, size, p_dec=0.3 if decimals else 0.0, exact=(n == "a")) for n in names}
        env, res = {n: dec(v) for n, v in vals.items()}, []
        for i, (x, op, y) in enumerate(steps, 1):
            env[f"r{i}"] = env[x] + env[y] if op == "+" else env[x] - env[y]
            res.append(env[f"r{i}"])
        if min(res) >= 0:
            break
    else:
        raise RuntimeError("no non-negative word problem found")
    user = []
    for piece in re.split(r"(\{\w\})", text):
        user.append(vals[piece[1]] if re.fullmatch(r"\{\w\}", piece) else piece)
    asst = []
    for i, (x, op, y) in enumerate(steps, 1):
        asst += [Num(dstr(env[x])), f" {op} ", Num(dstr(env[y])), " = ", Num(dstr(env[f"r{i}"])), "\n"]
    final = Num(dstr(res[-1]))
    asst += ["#### ", final]
    question = "".join(str(p) for p in user)
    return dict(task="wp_cot", size=size, user=user + [GSM_COT], asst=asst, answer=str(final), question=question)


MAKERS["wp_cot"] = ex_wp_cot


# ------------------------------------------------------------------------------------- GSM-Ranges
RANGES_REPO = "guactastesgood/GSM-Ranges"      # Shrestha, Kim and Ross (2025), arXiv:2502.08680; MIT licence
RANGES_KEEP = 10                               # variants kept per base question and level (the files hold 50)


def load_ranges(data_dir, log=print):
    """GSM-Ranges: 100 GSM8K test problems whose numbers are replaced by numbers of six ranges (level 1: the
    same number of digits, level 2: 100 to 1,000, ..., level 6: 1,000,000 to 10,000,000), in the format of
    GSM8K. Downloaded once from Hugging Face; the processed examples are cached as JSON in data_dir.
    Returns {level: [examples]} with level 0 = the original problems; the examples of a level are ordered by
    variant and then by base question, so that the first 100 * k examples are k variants of every question."""
    path = os.path.join(data_dir, "gsm_ranges_processed.json")
    if os.path.exists(path):
        raw = json.load(open(path, encoding="utf-8"))
    else:
        import csv
        from huggingface_hub import hf_hub_download
        os.makedirs(data_dir, exist_ok=True)
        raw, stats = {}, {}
        for level in range(7):
            name = "original.csv" if level == 0 else f"level{level}.csv"
            with open(hf_hub_download(RANGES_REPO, name, repo_type="dataset"), encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            seen, keep, digits = {}, [], []
            for r in rows:
                v = seen[r["index"]] = seen.get(r["index"], -1) + 1
                if v >= RANGES_KEEP:
                    continue
                ex = gsm_example(r["question"], r["answer"], max_steps=99)
                if ex is None:
                    continue
                ex.update(task="ranges", size=level, base=int(r["index"]), variant=v)
                digits += [sum(n_parts(p)) for p in ex["user"] + ex["asst"] if isinstance(p, Num)]
                keep.append(ex)
            keep.sort(key=lambda e: (e["variant"], e["base"]))
            for ex in keep:
                ex["user"], ex["asst"] = pack(ex["user"]), pack(ex["asst"])
            raw[str(level)] = keep
            stats[str(level)] = {"rows": len(rows), "questions": len(seen), "kept": len(keep),
                                 "mean_digits": round(sum(digits) / max(1, len(digits)), 2), "max_digits": max(digits, default=0)}
        raw["stats"] = stats
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(raw, f)
        os.replace(path + ".tmp", path)
        log(f"GSM-Ranges downloaded and processed: {json.dumps(stats)}")
    out = {level: [dict(r, user=unpack(r["user"]), asst=unpack(r["asst"])) for r in raw[str(level)]] for level in range(7)}
    out["stats"] = raw["stats"]
    return out


# ------------------------------------------------------------------------------------- rendering and scoring
def render(parts, mode="text"):
    """parts -> (text, []): the numbers written as digits."""
    return "".join(str(p) for p in parts), []


WORD_TASKS = ("gsm", "ranges", "wp", "domain")         # word problems (each also has a '<task>_cot' variant)
DOMAIN_TASKS = ("domain",)                             # word problems from various domains (domain_problems.py)


def base_task(task):
    return task[:-4] if task.endswith("_cot") else task


def score(ex, text):
    """Exact match of the final answer of an example with the generated text (numbers written as digits)."""
    if ex["task"] == "cmp":
        m = re.search(r"\b([AB])\b", text or "")
        return bool(m) and m.group(1) == ex["answer"]
    if base_task(ex["task"]) in DOMAIN_TASKS:          # equal, or rounds to the answer at its decimal places
        import domain_problems
        t = text.split("####")[-1] if "####" in (text or "") else text
        return domain_problems.correct(last_number(t), ex["answer"], ex.get("decimals"), ex.get("answer_alt"))
    if base_task(ex["task"]) in WORD_TASKS:            # the value counts, not the way it is written
        t = text.split("####")[-1] if "####" in (text or "") else text
        p = last_number(t)
        return p is not None and dec(p) == dec(ex["answer"])
    p = last_number(text)
    return p is not None and p == ex["answer"]


# ------------------------------------------------------------------------------------- training mixture / test sets
class Mixture:
    """Training stream: tasks drawn with the given weights; the size limit grows linearly from 3 to max_size
    during the first `ramp` fraction of the stage (length curriculum)."""

    def __init__(self, weights, max_size, seed, gsm_train=None, ramp=0.5, gsm_cot_train=None, domain_cot_train=None):
        self.tasks = [t for t, w in weights.items() if w > 0 and (t != "gsm" or gsm_train)
                      and (t != "gsm_cot" or gsm_cot_train) and (t != "domain_cot" or domain_cot_train)]
        self.w = [weights[t] for t in self.tasks]
        self.max_size, self.ramp = max_size, ramp
        self.rng = random.Random(seed)
        self.gsm = [r for r in (gsm_train or []) if r["trainable"]]
        self.gsm_cot = list(gsm_cot_train or [])
        self.domain_cot = list(domain_cot_train or [])

    def sample(self, progress):
        task = self.rng.choices(self.tasks, self.w)[0]
        if task == "gsm":
            return self.rng.choice(self.gsm)
        if task == "gsm_cot":
            return self.rng.choice(self.gsm_cot)
        if task == "domain_cot":                       # the model's own solutions of word problems from various domains
            return self.rng.choice(self.domain_cot)
        top = min(self.max_size, MAX_SIZE.get(task, self.max_size))
        cur = max(1, min(top, round(3 + (top - 3) * min(1.0, progress / self.ramp)))) if self.ramp > 0 else top
        return make(task, self.rng, self.rng.randint(1, cur))      # chain: digits of the terms; 2-4 steps

    def batch(self, n, progress):
        return [self.sample(progress) for _ in range(n)]


def test_cells(cfg, quick=False, ranges=None):
    """Fixed test sets: {task: {size: [examples]}}. Sizes above the training range are the 'beyond' cells.
    The key of a GSM-Ranges cell is its level (0 = the original problems). The word problems (templated and
    GSM-Ranges) are asked for a step-by-step solution (tasks wp_cot, ranges_cot). quick: a smaller set
    (24 problems per cell)."""
    cells = _test_cells(cfg, quick, ranges)
    n = cfg.get("eval_n_quick", 24) if quick else cfg["eval_n"]
    wrng = random.Random(54321)
    sizes = sorted(cells["wp"])
    cells.pop("wp")
    cells["wp_cot"] = {s: [ex_wp_cot(wrng, s, "test") for _ in range(n)] for s in sizes}
    if "ranges" in cells:
        cells["ranges_cot"] = {lv: [cot_parts(r) for r in exs] for lv, exs in cells.pop("ranges").items()}
    return cells


def _test_cells(cfg, quick=False, ranges=None):
    rng = random.Random(12345)
    n = cfg.get("eval_n_quick", 24) if quick else cfg["eval_n"]
    top = cfg["max_size"]
    sizes = [s for s in (1, 2, 3, 4, 6, 8, 10, 12) if s <= top] + ([] if quick else [top + 2, top + 4, top + 8])
    sizes = [s for s in sizes if s <= 22]
    if quick:
        sizes = [s for s in sizes if s in (2, 4, 8, 12)]
    cells = {}
    for task in ("add", "sub", "cmp"):
        cells[task] = {s: [make(task, rng, s) for _ in range(n)] for s in sizes}
    msz = [s for s in range(1, MAX_SIZE["mul"] + 1)] + ([] if quick else [MAX_SIZE["mul"] + 1, MAX_SIZE["mul"] + 2])
    cells["mul"] = {s: [make("mul", rng, s) for _ in range(n)] for s in (msz if not quick else (2, 4, 6))}
    cells["chain"] = {k: [ex_chain(rng, min(6, top), steps=k) for _ in range(n)] for k in ((2, 4) if quick else (2, 3, 4, 6))}
    wrng = random.Random(54321)                        # its own stream: the same problems for every method
    cells["wp"] = {s: [ex_wp(wrng, s, "test") for _ in range(n)] for s in ((4, 12) if quick else sizes) if s <= top or not quick}
    if ranges:
        if quick:
            cells["ranges"] = {lv: ranges[lv][:cfg.get("ranges_n_quick", 40)] for lv in (2, 5)}
        else:
            cells["ranges"] = {lv: ranges[lv][:cfg["ranges_n"]] for lv in range(7)}
    return cells


def in_range(task, size, cfg):
    """Is the test cell (task, size) inside the range seen in training?"""
    task = base_task(task)
    if task in ("gsm", "ranges") or task in DOMAIN_TASKS:
        return True
    if task == "chain":                                # the cell key of a chain is its number of steps
        return size <= CHAIN_TRAIN_STEPS
    return size <= min(cfg["max_size"], MAX_SIZE.get(task, cfg["max_size"]))
