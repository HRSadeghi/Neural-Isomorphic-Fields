"""Train one operator alone (Section 5.3.2, Figure 7, Table A.2). The operands are given as fixed one-hot
digit slots, and a linear head reads the digits of the result; there is no encoder and no decoder.

    python train_operator.py --name abl_x_full --op x --kind staged --steps 8000 --patience 6 --min-delta 0.05 --min-steps 1000

Options: --kind staged|coupled|dense, --op x|+, --layers k, --no-hints, --no-refit,
--fourier decimal|random|none, --soft-counts, --bf16 (configs/experiments.json lists the runs of the paper).

Run folder runs/<name>/: config.json, metrics.jsonl (every 50 steps), evals.jsonl (every 250 steps: held-out
exact match overall and by result length, and for the staged operator the accuracy of every stage),
ckpt/step_XXXXX.pt, last.pt, final.pt, best.pt, DONE.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from nif.convergence import convergence_state
from nif.layers import ANO, DenseANO, MulHintHeads, StagedMulANO, param_groups, refit_pair_front
from nif.numdata import E_MIN, K, admissible_pairs, canonical_sign, mul_hints, num_digits

HERE = os.path.dirname(os.path.abspath(__file__))

C = 16


def slots(sign, digits):
    x = torch.zeros(digits.shape[0], K, C, device=digits.device)
    x[..., :10] = F.one_hot(digits, 10).float()
    x[..., 10] = sign[:, None].float()
    return x.view(digits.shape[0], -1)



class OpProbe(nn.Module):
    def __init__(self, a):
        super().__init__()
        self.kind, self.op = a.kind, a.op
        fp32 = not a.bf16
        if a.kind == "staged":
            self.ano = StagedMulANO(K, C, 32, 192, 6, (2, 2, 2, 2), shift=E_MIN, fp32=fp32,
                                    front_kw=dict(soft_counts=a.soft_counts, fourier=a.fourier, refit=not a.no_refit))
            self.hints = MulHintHeads(192, 4)
        elif a.kind == "coupled":
            self.ano = ANO(K, C, 192, 6, a.layers or 6, 1, "coupled" if a.op == "x" else None, 32, 0.25, shift=E_MIN,
                           fp32=fp32)
        else:
            self.ano = DenseANO(K, C, a.layers or 4, fp32=fp32)
        self.head = nn.Linear(C, 10)
        self.sign = nn.Linear(C, 1)

    def forward(self, xa, xb, tokens=False):
        if self.kind == "staged":
            z, toks, aux = self.ano(xa, xb, return_tokens=True)
        else:
            z, toks, aux = self.ano(xa, xb), None, None
        z = z.view(-1, K, C)
        out = (self.head(z), self.sign(z.mean(1)).squeeze(-1))
        return (out, toks, aux) if tokens else out


@torch.no_grad()
def evaluate(model, op, n, dev, seed=999):
    model.eval()
    g = torch.Generator(device=dev).manual_seed(seed)
    sa, da, sb, db, sr, dr = admissible_pairs(n, op, dev, g, p_uniform=0.0)      # the paper's sampler
    sr = canonical_sign(sr, dr)
    oks, stages = [], {}
    for i in range(0, n, 1024):
        sl = slice(i, i + 1024)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            (ld, ls), toks, aux = model(slots(sa[sl], da[sl]), slots(sb[sl], db[sl]), tokens=True)
        nz = (dr[sl] != 0).any(1)
        oks.append((ld.float().argmax(-1) == dr[sl]).all(1) & ((torch.where(ls.float() > 0, 1, -1) == sr[sl]) | ~nz))
        if toks is not None and op == "x" and i == 0:                       # stage accuracies (first chunk)
            h = mul_hints(da[sl], db[sl])
            B = da[sl].shape[0]
            live = h["LH"].sum(-1) > 0
            acc = lambda p, y: round(100 * (p == y)[live].float().mean().item(), 2)
            stages["digits"] = round(100 * (aux[:B, :, :10].argmax(-1) == da[sl]).float().mean().item(), 2)
            stages["LH_within_0.5"] = round(100 * ((aux[:B, :, 20:22] - h["LH"].float()).abs() < 0.5).all(-1)[live]
                                            .float().mean().item(), 2)
            P = model.hints.planes(toks[0][:B]).view(B, K, 6, 10).argmax(-1)
            stages["planes"] = [acc(P[..., j], h["planes"][..., j]) for j in range(6)]
            rq = model.hints.rq(toks[1][:B])
            stages["S_mod10"] = acc(rq[..., :10].argmax(-1), h["rq"][..., 0])
            stages["T"] = acc(model.hints.T(toks[2][:B]).argmax(-1), h["T"].clamp(max=12))
    ok = torch.cat(oks)
    L = num_digits(sr, dr)[0]
    by = {f"{lo}-{hi}": round(100 * ok[(L >= lo) & (L <= hi)].float().mean().item(), 2)
          for lo, hi in [(1, 5), (6, 10), (11, 15), (16, 20), (21, 25)] if ((L >= lo) & (L <= hi)).any()}
    model.train()
    return {"em": round(100 * ok.float().mean().item(), 3), "by_len": by, "stages": stages}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--op", default="x", choices=["+", "x"])
    ap.add_argument("--kind", default="staged", choices=["staged", "coupled", "dense"])
    ap.add_argument("--no-hints", action="store_true")
    ap.add_argument("--no-refit", action="store_true")
    ap.add_argument("--fourier", default="decimal", choices=["decimal", "random", "none"])
    ap.add_argument("--soft-counts", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--layers", type=int, default=None, help="number of AC layers (dense: 4, coupled: 6)")
    ap.add_argument("--patience", type=int, default=0, help="evaluations without improvement before stopping (0: off)")
    ap.add_argument("--min-delta", type=float, default=0.05)
    ap.add_argument("--min-steps", type=int, default=0)
    ap.add_argument("--saturate", type=float, default=99.95, help="stop when the EM is at least this twice in a row")
    a = ap.parse_args()
    run = os.path.join(HERE, "runs", a.name)
    os.makedirs(os.path.join(run, "ckpt"), exist_ok=True)
    json.dump(vars(a), open(os.path.join(run, "config.json"), "w"), indent=2)
    if os.path.exists(os.path.join(run, "DONE")):
        print("DONE (already finished)", flush=True)
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda"
    torch.manual_seed(a.seed)
    g = torch.Generator(device=dev).manual_seed(a.seed)
    model = OpProbe(a).to(dev)
    opt = torch.optim.AdamW(param_groups(model, a.lr), betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    step = 0
    last = os.path.join(run, "last.pt")
    if os.path.exists(last):
        st = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        g.set_state(st["g"].cpu()); step = st["step"]
        print(f"resumed at step {step}", flush=True)
    print("parameters:", sum(p.numel() for p in model.parameters()), flush=True)

    def log(fname, rec):
        with open(os.path.join(run, fname), "a") as f:
            f.write(json.dumps(rec) + "\n")

    def save(path):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "g": g.get_state(), "step": step, "args": vars(a)}, path + ".tmp")
        os.replace(path + ".tmp", path)

    evaluated = {json.loads(l)["step"] for l in open(os.path.join(run, "evals.jsonl"))} \
        if os.path.exists(os.path.join(run, "evals.jsonl")) else set()
    t0 = time.time()
    hinted = a.kind == "staged" and not a.no_hints
    stop_reason = f"step cap {a.steps} reached"
    if os.path.exists(os.path.join(run, "evals.jsonl")):      # a resume right after the stopping evaluation
        _ev = [json.loads(l) for l in open(os.path.join(run, "evals.jsonl"))]
        if _ev and _ev[-1]["step"] == step and _ev[-1].get("convergence", {}).get("stop"):
            stop_reason = f"converged at step {step}: {_ev[-1]['convergence']['reason']}"
            step = a.steps + 1
    while step <= a.steps:
        if (step % a.eval_every == 0 or step == a.steps) and step not in evaluated:
            ev = evaluate(model, a.op, 4096, dev)
            ev.update(step=step, wall=round(time.time() - t0, 1))
            prev = [json.loads(l) for l in open(os.path.join(run, "evals.jsonl"))] \
                if os.path.exists(os.path.join(run, "evals.jsonl")) else []
            stop, conv = convergence_state(prev + [ev], "em", a.min_delta, a.patience, a.min_steps, a.saturate)
            ev["convergence"] = conv
            log("evals.jsonl", ev)
            evaluated.add(step)
            print(f"[eval] step {step}: {json.dumps(ev)}", flush=True)
            if stop:
                stop_reason = f"converged at step {step}: {conv['reason']}"
                print("[convergence] " + stop_reason, flush=True)
                break
        if step == a.steps:
            break
        opt.zero_grad(set_to_none=True)
        sa, da, sb, db, sr, dr = admissible_pairs(a.batch, a.op, dev, g, p_uniform=0.5)
        sr = canonical_sign(sr, dr)
        xa, xb = slots(sa, da), slots(sb, db)
        if a.kind == "staged" and not a.no_refit and step > 0 and step % (100 if step <= 1000 else 500) == 0:
            refit_pair_front(model.ano, xa, xb, mul_hints(da, db)["LH"], opt)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            (ld, ls), toks, aux = model(xa, xb, tokens=True)
        loss_main = F.cross_entropy(ld.float().reshape(-1, 10), dr.reshape(-1)) + \
            F.binary_cross_entropy_with_logits(ls.float(), (sr > 0).float())
        loss = loss_main
        hint = torch.zeros((), device=dev)
        if hinted:
            hint = model.hints.loss(toks, aux, mul_hints(da, db), dr, sr, da, db,
                                    canonical_sign(sa, da), canonical_sign(sb, db))
            loss = loss + hint
        loss.backward()
        norms = torch.stack([p.grad.detach().double().norm() for p in model.parameters() if p.grad is not None])
        gn = norms.norm()
        sc = min(1.0, 1.0 / (gn.item() + 1e-6))
        if sc < 1:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(sc)
        opt.step()
        sched.step()
        step += 1
        if step % 50 == 0:
            log("metrics.jsonl", {"step": step, "loss": round(loss_main.item(), 5), "hint": round(hint.item(), 5),
                                  "gn": round(gn.item(), 3), "lr": sched.get_last_lr()[0], "wall": round(time.time() - t0, 1)})
        if step % 25 == 0:
            save(last)
        if step % a.ckpt_every == 0:
            save(os.path.join(run, "ckpt", f"step_{step:05d}.pt"))
    save(last)
    torch.save({"model": model.state_dict(), "args": vars(a), "step": step}, os.path.join(run, "final.pt"))
    evs = [json.loads(l) for l in open(os.path.join(run, "evals.jsonl"))]
    best = max((e for e in evs if os.path.exists(os.path.join(run, "ckpt", f"step_{e['step']:05d}.pt")) or e["step"] == step),
               key=lambda e: e["em"])
    src = os.path.join(run, "ckpt", f"step_{best['step']:05d}.pt")
    sd = torch.load(src, map_location="cpu", weights_only=False)["model"] if os.path.exists(src) else model.state_dict()
    torch.save({"model": sd, "args": vars(a), "step": best["step"], "em": best["em"]}, os.path.join(run, "best.pt"))
    print(f"best checkpoint: step {best['step']} (EM {best['em']})", flush=True)
    open(os.path.join(run, "DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S") + " " + stop_reason + "\n")
    print("DONE", stop_reason, flush=True)


if __name__ == "__main__":
    main()
