"""Train and test a language model with NIF operator layers (Section 5.4 of the paper), or test the base model.

    python downstream/train.py --method op_nif --out downstream/runs/operator_layers
    python downstream/train.py --method zeroshot --out downstream/runs/base_model
    python downstream/train.py --method op_nif --out downstream/runs/test_6000 --set init_ckpt=<checkpoint> steps=6000

Methods
    zeroshot  the language model as released; numbers written as digits; no training
    op_nif    operator layers (oplayers.py): numbers as digits + a workspace of tagged value tokens after the question;
              the frozen NIF encoder, operators and decoder run inside two layers of the language model

The settings of the runs of the paper are in downstream/run.py. A run that starts from a checkpoint whose step equals
`steps` only runs the tests. Files written to the run folder:
    config.json, log.txt, status.json, metrics.jsonl (training), evals.jsonl (tests during training),
    cells_steps.jsonl and items_step<N>.jsonl.gz (the complete test every eval_every steps, every output),
    results.json (final test and interventions), items.jsonl.gz (every test problem and output),
    causal_items.jsonl.gz (outcomes with the operator layers switched off / with wrong vectors), DONE;
    ckpt.pt (weights, optimizer, data streams) in --ckpt-dir
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import torch  # noqa: E402

import data as D  # noqa: E402
from nif.paths import checkpoint  # noqa: E402

DEFAULTS = dict(
    model="Qwen/Qwen3.5-0.8B",
    nif_ckpt=checkpoint("main_s0"),
    data_dir=os.path.join(HERE, "data"),
    max_size=12,                # integer digits of the operands in training
    steps=6000,
    batch=32, warmup=100, clip=1.0,
    full_ft=True,               # train every weight of the language model (float32 master weights)
    lr_full=1e-5,               # learning rate of the language model
    lr_op=1e-3,                 # learning rate of the adapters of the operator layers
    accum=2,                    # micro-batches per step (gradients added; the same update with less GPU memory)
    # training mixture: synthetic arithmetic, the model's own step-by-step solutions of GSM8K training problems and of
    # word problems from various domains, templated word problems with step-by-step targets
    mix=dict(add=0.1, sub=0.1, chain=0.05, mul=0.12, cmp=0.03, gsm_cot=0.15, wp_cot=0.15, domain_cot=0.3),
    gsm_solutions="",           # default <data_dir>/gsm8k_solutions.json.gz (make_solutions.py --source gsm8k)
    domain_solutions="",        # default <data_dir>/domain_solutions.json.gz (make_solutions.py --source domains)
    sanity=True,                # stop if the loaded language model does not model plain English
    log_every=25, ckpt_every=500, eval_every=500,
    eval_n=100,                 # test problems per cell
    ranges_n=200,               # GSM-Ranges problems per level (2 variants of each of the 100 questions)
    max_new_text=96, max_new_cot=512,
    eval_batch=128,             # prompts per decoding batch (word problems: half)
    # speed (none of these changes what is computed, except the order of the examples with group > 1)
    dtype="auto",               # frozen model: bfloat16 on the GPU, float32 on the CPU (full_ft: float32 + autocast)
    kernels="fla",              # linear-attention layers: "fla" (Triton kernels if installed) or "torch" (reference)
    group=8,                    # draw group * batch examples, sort them by length, cut them into batches (less padding)
    compile=False,              # torch.compile of the decoder for training
    # operator layers (downstream/oplayers.py)
    op_layers="8,14",           # decoder layers (0-based) after which an operator layer runs; each has its own adapters
    op_max_spans=12,            # numbers of a prompt that get a workspace entry (the first ones)
    op_key=256,                 # width of the pointer queries and keys
    op_writeback=True,          # inverse adapters write sum / number / product into the hidden states
    op_copy=True,               # the answer is written by the mixture of the language model and the registers
    op_dynamic=32,              # numbers of the answer that get registers (operations on generated numbers)
    op_window=3,                # distant supervision: a result must be one of the next op_window numbers of the answer
    w_ptr=1.0,                  # weight of the pointer loss of the operator layers
    w_lm=1.0,                   # weight of the language model's own loss next to the mixture
    causal_n=200,               # final test: problems per cell for the tests without operator layers / with wrong vectors
    # continuing a run: start from its checkpoint (weights; optimizer and data streams if present) and train on to
    # `steps` with a second learning-rate cycle from restart_at: warm-up from 10% to restart_peak of the base rate in
    # `warmup` steps, then cosine decay to 10% at the end
    init_ckpt="",               # checkpoint that the run starts from (used only if this run has no checkpoint of its own)
    restart_at=0,               # step at which the second cycle starts (0: one cycle)
    restart_peak=0.4,           # peak of the second cycle (fraction of the base learning rate)
)

# the test cells reported in the paper (Table 7, Figures 11 and 12); None: every size
REPORTED = {"add": None, "sub": None, "mul": None, "wp_cot": None, "ranges_cot": (1, 2, 3, 4, 5, 6)}


def lr_scale_of(s, cfg, steps):
    """Learning-rate factor of step s: warm-up, then cosine decay to 10%; with restart_at > 0 a second cycle (see
    DEFAULTS) from step restart_at."""
    r = cfg.get("restart_at", 0)
    if r and s >= r:
        k, m, peak = s - r, steps - r, cfg.get("restart_peak", 0.4)
        if k < cfg["warmup"]:
            return 0.1 + (peak - 0.1) * (k + 1) / cfg["warmup"]
        return 0.1 + (peak - 0.1) * 0.5 * (1 + math.cos(math.pi * (k - cfg["warmup"]) / max(1, m - cfg["warmup"])))
    if s < cfg["warmup"]:
        return (s + 1) / cfg["warmup"]
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - cfg["warmup"]) / max(1, steps - cfg["warmup"])))


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["zeroshot", "op_nif"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt-dir", default=None, help="folder of ckpt.pt (default: the run folder)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = dict(DEFAULTS)
    for kv in a.set:
        k, v = kv.split("=", 1)
        if k not in cfg:
            raise SystemExit(f"unknown setting {k}")
        old = cfg[k]
        if isinstance(old, bool):
            cfg[k] = v.lower() in ("1", "true", "yes")
        elif isinstance(old, int):
            cfg[k] = int(float(v))
        elif isinstance(old, float):
            cfg[k] = float(v)
        elif isinstance(old, dict):
            cfg[k] = json.loads(v)
        else:
            cfg[k] = v
    if a.method == "zeroshot":                                  # nothing is trained
        cfg.update(steps=0, full_ft=False)
    return a, cfg


class Run:
    """Run folder: log, heartbeat and JSON lines."""

    def __init__(self, out, method, seed, device="cpu"):
        self.out, self.t0 = out, time.time()
        try:
            self.gpu = torch.cuda.get_device_name(0) if str(device).startswith("cuda") else "cpu"
        except Exception:
            self.gpu = str(device)
        os.makedirs(out, exist_ok=True)
        self.base = {"job": f"{method}_s{seed}", "method": method, "seed": seed}
        self.state = {}

    def log(self, msg):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        with open(os.path.join(self.out, "log.txt"), "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def status(self, **kw):
        self.state.update(kw)
        s = dict(self.base, **self.state, time=time.time(), gpu=self.gpu)
        tmp = os.path.join(self.out, "status.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(s, f)
        os.replace(tmp, os.path.join(self.out, "status.json"))

    def append(self, name, row):
        with open(os.path.join(self.out, name), "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def path(self, name):
        return os.path.join(self.out, name)


# ------------------------------------------------------------------------------------- evaluation
def is_oom(e):
    """Out of GPU memory. On a MIG slice whose memory NVML may not read, PyTorch can report it as an internal assert
    of the caching allocator instead of OutOfMemoryError."""
    msg = str(e)
    return isinstance(e, torch.OutOfMemoryError) or "out of memory" in msg.lower() or "NVML_SUCCESS" in msg \
        or "CUDACachingAllocator" in msg


def generate_safe(llm, examples, max_new):
    """llm.generate; if the GPU runs out of memory (e.g. another job on the same GPU), the batch is decoded in two
    halves (greedy decoding: the same outputs). Returns (outputs, register strings of the operator layers or None)."""
    log0 = dict(llm.op_log) if getattr(llm, "op", None) is not None else None
    try:
        return llm.generate(examples, max_new), getattr(llm, "op_regs", None)
    except (torch.OutOfMemoryError, RuntimeError) as e:
        if len(examples) == 1 or not is_oom(e):
            raise
        if log0 is not None:
            llm.op_log = log0                                # the statistics of the failed attempt do not count
        torch.cuda.empty_cache()
        h = len(examples) // 2
        g1, r1 = generate_safe(llm, examples[:h], max_new)
        g2, r2 = generate_safe(llm, examples[h:], max_new)
        return g1 + g2, (r1 + r2 if r1 is not None and r2 is not None else None)


def length_buckets(llm, examples, batch):
    """Batches of examples with prompts of similar length (sorted by length, so that little padding is needed)."""
    lens = [len(r[0]) for r in llm.encode_examples(examples, with_answer=False)]
    order = sorted(range(len(examples)), key=lambda i: lens[i])
    for j in range(0, len(order), batch):
        yield order[j:j + batch]


def max_new_of(task, cfg):
    return cfg["max_new_cot"] if task.endswith("_cot") else cfg["max_new_text"]


def batch_of(task, cfg):
    return max(1, cfg["eval_batch"] // 2) if D.base_task(task) in D.WORD_TASKS else cfg["eval_batch"]


@torch.no_grad()
def evaluate(llm, cells, cfg, run):
    """Exact match of the final answer in every test cell. Returns (cells, summary, samples, diagnostics, items);
    items: every test problem with its output."""
    llm.eval()
    opm = llm.op is not None
    res, samples, items = {}, {}, []
    t0 = time.time()
    for task, by_size in cells.items():
        res[task] = {}
        for size, exs in by_size.items():
            ok, outs = [None] * len(exs), [None] * len(exs)
            regs = [None] * len(exs)                          # operator layers: the registers of every problem
            t_cell = time.time()
            if opm:
                llm.op_log = dict.fromkeys(llm.op_log, 0)
            for idx in length_buckets(llm, exs, batch_of(task, cfg)):
                gen, got_regs = generate_safe(llm, [exs[i] for i in idx], max_new_of(task, cfg))
                got_regs = got_regs if opm else None
                for k, (i, (text, _)) in enumerate(zip(idx, gen)):
                    ok[i], outs[i] = D.score(exs[i], text), text
                    if got_regs is not None:
                        regs[i] = got_regs[k]
                run.status(phase="test", eval_cell=f"{task}/{size}")
            res[task][str(size)] = {"n": len(exs), "em": round(100.0 * sum(ok) / max(1, len(exs)), 2),
                                    "sec": round(time.time() - t_cell, 2)}
            if opm and llm.op_log["prompts"]:                  # how much of the output the registers wrote
                lg = llm.op_log
                res[task][str(size)].update(copied=round(100.0 * lg["copied"] / max(1, lg["tokens"]), 2),
                                            registers=round(lg["registers"] / lg["prompts"], 2))
                hit = [r is not None and exs[i]["answer"] in r["results"] for i, r in enumerate(regs)]
                res[task][str(size)]["answer_in_results"] = round(100.0 * sum(hit) / max(1, len(exs)), 2)
            items += [dict({"task": task, "size": size, "i": i, "prompt": D.render(exs[i]["user"])[0],
                            "target": exs[i]["answer"], "output": outs[i], "ok": bool(ok[i])},
                           **({"registers": regs[i]} if regs[i] is not None else {})) for i in range(len(exs))]
            wrong = [i for i, o in enumerate(ok) if not o][:3]
            right = [i for i, o in enumerate(ok) if o][:1]
            samples[f"{task}/{size}"] = [dict({"prompt": D.render(exs[i]["user"])[0][-400:], "target": exs[i]["answer"],
                                               "output": outs[i][-300:], "ok": bool(ok[i])},
                                              **({"registers": regs[i]} if regs[i] is not None else {})) for i in wrong + right]
    summary = {}
    for task, by_size in res.items():
        if D.base_task(task) == "ranges":              # one value per level: ranges_cot_L1 ... ranges_cot_L6
            for s, v in by_size.items():
                summary[f"{task}_L{s}"] = v["em"]
            continue
        for label, keep in (("in", True), ("beyond", False)):
            sel = [(v["em"], v["n"]) for s, v in by_size.items() if D.in_range(task, int(s), cfg) == keep]
            if sel:
                summary[f"{task}_{label}"] = round(sum(e * n for e, n in sel) / sum(n for _, n in sel), 2)
    return res, summary, samples, {"seconds": round(time.time() - t0, 1)}, items


@torch.no_grad()
def causal_tests(llm, cells, cfg, run):
    """Operator layers: exact match of the trained model on the same problems (a) as trained, (b) with the operator
    layers and the mixture switched off (the language model alone writes the answer), (c) with the vectors of
    other numbers in the workspace. The problems are batched as in evaluate(), so (a) equals the final test."""
    llm.eval()
    out, items = {}, []
    for task in REPORTED:
        for size, exs in cells.get(task, {}).items():
            exs = exs[:cfg["causal_n"]]
            if not exs:
                continue
            row, oks = {"n": len(exs)}, {}
            for mode in ("on", "off", "shuffle"):
                llm.op_mode = mode
                ok = [False] * len(exs)
                for idx in length_buckets(llm, exs, batch_of(task, cfg)):
                    gen, _ = generate_safe(llm, [exs[i] for i in idx], max_new_of(task, cfg))
                    for i, (text, _) in zip(idx, gen):
                        ok[i] = bool(D.score(exs[i], text))
                row[mode] = round(100.0 * sum(ok) / len(exs), 2)
                oks[mode] = ok
            llm.op_mode = "on"
            out.setdefault(task, {})[str(size)] = row
            items += [dict({"task": task, "size": size, "i": i}, **{m: oks[m][i] for m in oks}) for i in range(len(exs))]
            run.status(phase="interventions", eval_cell=f"{task}/{size}")
    return out, items


def reported(cells):
    """The test cells of the paper. All cells are generated (they share one random stream), then the others dropped."""
    return {t: {s: v for s, v in by.items() if REPORTED[t] is None or s in REPORTED[t]}
            for t, by in cells.items() if t in REPORTED}


# ------------------------------------------------------------------------------------- main
def main():
    a, cfg = parse()
    run = Run(a.out, a.method, a.seed, a.device)
    ckpt = os.path.join(a.ckpt_dir or a.out, "ckpt.pt")
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    if os.path.exists(run.path("DONE")):
        run.log("already finished")
        return
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    json.dump(dict(cfg, method=a.method, seed=a.seed), open(run.path("config.json"), "w"), indent=1)
    run.log(f"job {a.method}_s{a.seed} on {a.device}; model {cfg['model']}")
    run.status(phase="loading", step=0, total=0)

    gsm_cot_train = domain_cot_train = None
    if cfg["steps"]:                                            # the model's own solutions (make_solutions.py)
        path = cfg["gsm_solutions"] or os.path.join(cfg["data_dir"], "gsm8k_solutions.json.gz")
        gsm_cot_train, st = D.load_self_cot(path)
        run.log(f"GSM8K training solutions written by the model itself: {len(gsm_cot_train)} from {path} ({json.dumps(st)})")
        path = cfg["domain_solutions"] or os.path.join(cfg["data_dir"], "domain_solutions.json.gz")
        domain_cot_train, st = D.load_self_cot(path)
        run.log(f"solutions of word problems from various domains: {len(domain_cot_train)} from {path} ({json.dumps(st)})")
    cuda = str(a.device).startswith("cuda")
    os.environ["NIF_DS_KERNELS"] = cfg["kernels"]             # read by llm.py before transformers is imported
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        frac = os.environ.get("NIF_DS_MEM_FRACTION")            # several jobs on one GPU
        if frac:
            torch.cuda.set_per_process_memory_fraction(float(frac))
            run.log(f"GPU memory of this job limited to {float(frac):.0%} of the device (shared with other jobs); "
                    f"a batch that does not fit is processed in two halves")
    from llm import NumLLM, kernel_report
    dtype = {"auto": torch.bfloat16 if cuda else torch.float32, "bf16": torch.bfloat16, "fp32": torch.float32}[cfg["dtype"]]
    if cfg["full_ft"]:
        dtype = torch.float32                                   # master weights; the products run in bfloat16 (autocast)
    ops, op_cfg = None, None
    if a.method == "op_nif":                                    # operator layers
        import oplayers
        if not os.path.exists(cfg["nif_ckpt"]):
            raise SystemExit(f"op_nif: NIF checkpoint not found ({cfg['nif_ckpt']!r})")
        ops = oplayers.NIFAlgebra(oplayers.NIFNumbers(cfg["nif_ckpt"]))
        op_cfg = {"layers": [int(x) for x in str(cfg["op_layers"]).split(",") if x.strip() != ""],
                  "max_spans": cfg["op_max_spans"], "d_key": cfg["op_key"], "writeback": cfg["op_writeback"], "copy": cfg["op_copy"],
                  "dyn_max": cfg["op_dynamic"], "dyn_window": cfg["op_window"]}
    llm = NumLLM(cfg["model"], dtype=dtype, ops=ops, op_cfg=op_cfg, full_ft=cfg["full_ft"]).to(a.device)
    if ops is not None:
        llm.w_ptr = cfg["w_ptr"]
        llm.w_lm = cfg["w_lm"]
    n_train = sum(p.numel() for p in llm.interface_parameters()) + \
        (sum(p.numel() for p in llm.base_parameters()) if cfg["full_ft"] else 0)
    n_base = sum(p.numel() for n, p in llm.named_parameters() if not n.startswith(("op.", "opv.")))
    run.log(f"base model {n_base / 1e6:.0f}M parameters ({str(dtype).replace('torch.', '')}), trained {n_train / 1e6:.2f}M "
            f"({100 * n_train / n_base:.2f}%)" + (" - full fine-tuning of the language model" if cfg["full_ft"] else "")
            + (f"; operations on the first {cfg['op_dynamic']} numbers of every answer" if ops is not None and cfg["op_dynamic"] else "")
            + (f", operator layers after decoder layers {op_cfg['layers']} with the frozen {ops.name} model (dim {ops.dim}); "
               f"number characters that are single tokens: {''.join(sorted(llm.c2id))}" if ops else ""))
    kern = kernel_report()
    versions = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda}
    for mod in ("transformers", "triton", "fla"):
        try:
            versions[mod] = getattr(__import__(mod), "__version__", "?")
        except Exception:
            versions[mod] = None
    kern["versions"] = versions
    run.log(f"versions: {json.dumps(versions)}")
    run.log(f"linear-attention kernels: {json.dumps(kern)}; chat template rendered once: {llm._wrap is not None}")
    run.status(kernels=kern)

    nll, diff = llm.sanity_nll()
    run.log(f"loaded with {llm.loader}; English text: {nll:.2f} nats per token; decoder + output layer vs complete model: "
            f"largest relative difference of the logits {diff:.1e}")
    if cfg["sanity"] and nll > 6.0:
        raise SystemExit("the language model does not model plain English: its weights were not loaded correctly")
    if diff > 2e-2:
        raise SystemExit("the decoder + output layer used here does not reproduce the logits of the complete model")
    ranges = D.load_ranges(cfg["data_dir"], run.log)
    cells_quick = D.test_cells(cfg, quick=True, ranges=ranges)          # only for the check of the decoding cache
    cells_final = reported(D.test_cells(cfg, quick=False, ranges=ranges))
    run.log(llm.check_cache(cells_quick["add"][max(cells_quick["add"])][:4] + cells_quick["chain"][max(cells_quick["chain"])][:2]))

    total = cfg["steps"]
    mix = dict(cfg["mix"])
    streams = {2: D.Mixture(mix, cfg["max_size"], a.seed + 2, gsm_cot_train=gsm_cot_train,
                            domain_cot_train=domain_cot_train)}
    if total:
        run.log("training mixture: " + ", ".join(f"{t} {w:.2f}" for t, w in zip(streams[2].tasks, streams[2].w)))

    # loss of a fixed batch before training: the same number for the same seed whatever the GPU or the kernels
    if total:
        probe = D.Mixture(mix, cfg["max_size"], 999, gsm_cot_train=gsm_cot_train,
                          domain_cot_train=domain_cot_train).batch(min(16, cfg["batch"]), 1.0)
        llm.eval()
        with torch.no_grad():
            pl, pparts = llm.loss(llm.collate(probe, a.device))
        run.log(f"probe loss (fixed batch, before training): {pl.item():.5f} " + " ".join(f"{k} {float(v):.5f}" for k, v in pparts.items()))
        run.status(probe_loss=round(pl.item(), 6))

    opt, step, wall, queue = None, 0, 0.0, []
    if ops is not None:
        llm.set_trainable()
        groups = []
        if cfg["full_ft"]:
            groups.append({"params": llm.base_parameters(), "lr": cfg["lr_full"], "base": cfg["lr_full"], "kind": "full"})
        groups.append({"params": llm.interface_parameters(), "lr": cfg["lr_op"], "base": cfg["lr_op"], "kind": "op"})
        groups = [g for g in groups if g["params"]]
        opt = torch.optim.AdamW(groups, weight_decay=0.0, betas=(0.9, 0.99), **({"fused": True} if cuda else {}))
        src = ckpt if os.path.exists(ckpt) else cfg["init_ckpt"] if cfg["init_ckpt"] else None
        if src and not os.path.exists(src):
            raise SystemExit(f"init_ckpt not found: {src}")
        if src:
            if src != ckpt:
                run.log(f"starting from {src}")
            st = torch.load(src, map_location="cpu", weights_only=False)
            llm.load_trainable(st["state"])
            if "opt" in st:
                opt.load_state_dict(st["opt"])
            step, wall = st["step"], st.get("wall", 0.0)
            for k, s in st.get("streams", {}).items():
                streams[int(k)].rng.setstate(s)
            queue = st.get("queue", [])
            run.log(f"resumed from step {step}")
        if cfg["compile"] and cuda:
            llm.compile(run.log)

    def next_batch(progress):
        """The next training batch. With group > 1, group * batch examples are drawn at once, sorted by length and
        cut into batches that are used in random order: far less padding, the same examples."""
        nonlocal queue
        if cfg["group"] <= 1:
            return streams[2].batch(cfg["batch"], progress)
        if not queue or queue[0][0] != 2:
            pool = streams[2].batch(cfg["batch"] * cfg["group"], progress)
            pool.sort(key=lambda e: len(D.render(e["user"])[0]) + len(D.render(e["asst"])[0]))
            cut = [pool[i:i + cfg["batch"]] for i in range(0, len(pool), cfg["batch"])]
            streams[2].rng.shuffle(cut)
            queue = [(2, b) for b in cut]
        return queue.pop(0)[1]

    def save():
        tmp = ckpt + ".tmp"
        torch.save({"state": llm.trainable_state(), "opt": opt.state_dict(), "step": step, "wall": wall,
                    "streams": {k: s.rng.getstate() for k, s in streams.items()}, "queue": queue}, tmp)
        os.replace(tmp, ckpt)
        run.status(ckpt_step=step)

    t_last, acc, seen, ooms, ntok, npad = time.time(), {}, 0, 0, 0, 0
    while step < total:
        llm.train()
        llm.set_trainable()
        examples = next_batch(step / max(1, total))
        for g in opt.param_groups:
            g["lr"] = g["base"] * lr_scale_of(step, cfg, total)

        def run_parts(n_parts):
            """Loss and gradients of the batch in n_parts micro-batches (the gradients are added: the same update)."""
            tot, agg, ntk, npd = 0.0, {}, 0, 0
            size = -(-len(examples) // n_parts)
            for s0 in range(0, len(examples), size):
                part = examples[s0:s0 + size]
                b_ = llm.collate(part, a.device)
                l_, p_ = llm.loss(b_)
                ntk, npd = ntk + b_["tokens"], npd + b_["padded"]
                w = len(part) / len(examples)
                if torch.isfinite(l_):                  # one synchronisation per micro-batch
                    (l_ * w).backward()
                tot = tot + l_.detach() * w
                for k, v in p_.items():
                    agg[k] = agg.get(k, 0.0) + v * len(part) / len(examples)
            return tot, agg, ntk, npd

        n_parts = cfg["accum"]
        while True:
            try:
                loss, parts, tk, pd = run_parts(n_parts)
                break
            except (torch.OutOfMemoryError, RuntimeError) as e:    # a batch of long problems: smaller micro-batches
                if not is_oom(e) or n_parts >= len(examples):
                    raise
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                n_parts *= 2
                ooms += 1
                if ooms <= 5:
                    run.log(f"step {step}: out of GPU memory, batch processed in {n_parts} parts")
        ntok, npad = ntok + tk, npad + pd
        if not torch.isfinite(loss):
            run.log(f"step {step}: non-finite loss, batch skipped")
            opt.zero_grad(set_to_none=True)
            step += 1
            continue
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], cfg["clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        seen += 1
        for k, v in dict(parts, loss=loss.detach()).items():
            acc[k] = acc.get(k, 0.0) + v                # tensors on the device: read only when logged
        if step % cfg["log_every"] == 0 or step == total:
            now = time.time()
            wall += now - t_last
            sps = (now - t_last) / seen
            row = {"step": step, "wall": round(wall, 1), "s_per_step": round(sps, 3),
                   "tok_per_s": round(ntok / max(1e-9, now - t_last)), "pad_frac": round(1 - ntok / max(1, npad), 3),
                   **{k: round(float(v) / seen, 4) for k, v in acc.items()}}
            ntok = npad = 0
            run.append("metrics.jsonl", row)
            run.status(phase="training", step=step, total=total, loss=row["loss"], s_per_step=row["s_per_step"],
                       eta_min=round((total - step) * sps / 60, 1), wall=round(wall, 1))
            if step % (cfg["log_every"] * 4) == 0:
                run.log(f"step {step}/{total} loss {row['loss']:.4f} " + " ".join(
                    f"{k} {v:.3f}" for k, v in row.items() if k in ("lm", "mix", "ptr", "dptr", "dptr_acc", "ptr_acc",
                                                                    "copy_p", "copy_cov")) + f" {sps:.2f}s/step")
            t_last, acc, seen = time.time(), {}, 0
        if step % cfg["ckpt_every"] == 0 or step == total:
            save()
        if step % cfg["eval_every"] == 0 and step < total:     # the complete test; every output kept
            cells_s, summ, _, dg, items_s = evaluate(llm, cells_final, cfg, run)
            with gzip.open(run.path(f"items_step{step}.jsonl.gz"), "wt", encoding="utf-8") as f:
                for it in items_s:
                    f.write(json.dumps(it) + "\n")
            run.append("cells_steps.jsonl", {"step": step, "cells": cells_s})
            run.append("evals.jsonl", {"step": step, **summ, "eval_seconds": dg["seconds"]})
            run.log(f"[test] step {step}: " + " ".join(f"{k} {v}" for k, v in summ.items()))
            t_last = time.time()

    run.status(phase="final test", step=total, total=total)
    cells, summ, samples, dg, items = evaluate(llm, cells_final, cfg, run)
    with gzip.open(run.path("items.jsonl.gz"), "wt", encoding="utf-8") as f:     # every test problem and output
        for it in items:
            f.write(json.dumps(it) + "\n")
    causal = None
    if ops is not None and cfg["causal_n"] > 0:
        causal, causal_items = causal_tests(llm, cells_final, cfg, run)
        with gzip.open(run.path("causal_items.jsonl.gz"), "wt", encoding="utf-8") as f:
            for it in causal_items:
                f.write(json.dumps(it) + "\n")
        for task, by_size in causal.items():
            run.log(f"[interventions] {task}: " + "; ".join(f"{s}: as trained {v['on']}, without operator layers "
                                                            f"{v['off']}, wrong vectors {v['shuffle']}"
                                                            for s, v in by_size.items()))
    out = {"job": f"{a.method}_s{a.seed}", "method": a.method, "seed": a.seed, "model": cfg["model"], "cells": cells,
           "summary": summ, "diagnostics": dg, "samples": samples, "causal": causal,
           "train": {"steps": total, "hours": round(wall / 3600, 3), "trained_params": n_train, "base_params": n_base,
                     "decoding_cache": llm.use_cache, "dtype": str(dtype).replace("torch.", ""), "kernels": kern,
                     "compiled": llm.compiled is not None, "gpu": run.gpu, "probe_loss": run.state.get("probe_loss"),
                     "eval_hours": round(dg["seconds"] / 3600, 3),
                     "op": dict(op_cfg, model=ops.name, operators=list(ops.has),
                                gates=[[round(v, 4) for v in torch.tanh(s_.gate.detach()).tolist()] for s_ in llm.op.stages],
                                in_gate=round(float(torch.tanh(llm.op.in_gate.detach())), 4)) if ops is not None else None},
           "ranges_stats": ranges["stats"], "config": cfg}
    json.dump(out, open(run.path("results.json"), "w"), indent=1)
    run.append("evals.jsonl", {"step": total, "final": True, **summ, "eval_seconds": dg["seconds"]})
    run.log("[final] " + " ".join(f"{k} {v}" for k, v in summ.items()))
    open(run.path("DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    run.status(phase="finished", step=total, total=total)
    run.log("DONE")


if __name__ == "__main__":
    main()
