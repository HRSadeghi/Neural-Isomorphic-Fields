"""Operator layers: the trained operators of a number embedding inside the layers of a language model.

Prompt. The text of the problem stays as it is (numbers as digits). After it, a workspace lists every number of the
prompt as three tokens:  <|num_start|> <|num_val|> <|num_end|>.  The two tags are embedded by the language model
(trained embeddings); the value token is the vector of the number from the frozen number encoder, through an input
adapter. The workspace comes after the question, so its tokens see the whole problem (the model is causal).

Operator layer (after a chosen layer of the language model; each one has its own adapters). The numbers and the
results computed so far are kept as vectors of the number embedding ("registers"). For the j-th number, with value
vector x:
  * the state of its <|num_start|> tag is mapped by an adapter to a query that points at one register u (or at
    "nothing") and chooses u + y, u - y or y - u; the frozen addition operator computes the result;
  * the state of its <|num_end|> tag is mapped to a query that points at one register u (or at "nothing"); the
    frozen multiplication operator computes u * y;
  * y is the number x itself or a result that this number produced in an earlier operator layer (second pointer);
  * inverse adapters write the sum to the <|num_start|> position, the number to the value position and the product
    to the <|num_end|> position of the hidden states (gated, the gates start at zero), and the next layers go on.
The adapters never produce a vector of their own: an adapter output is a choice among vectors that the number
model produced, so every input of an operator is a valid embedding. The numbers are processed from left to right
inside one layer, so that a result is available to the numbers after it (chains need one layer).

Output. The language model writes the answer token by token as always. At every position a pointer (query from
the last hidden state) distributes probability over "the language model's own next token" and "the next character
of register r" for every register whose decoded string continues the number being written (pointer-generator
mixture). The register strings come from the decoder of the number model.

Training. No gradient passes through the number model. The pointers of the operator layers are trained by distant
supervision: from the numbers of the prompt and the numbers of the target text alone, build_program() finds which
(operand, own value, operation) produce a number of the target; the loss is the negative log of the total
probability of these choices (or of "nothing" if there is none), and these registers are the ones kept (teacher
forcing). The output pointer is trained by the likelihood of the mixture. At test time every choice is the arg max
of its pointer, and no target is used anywhere.
"""
from __future__ import annotations

import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import canon, dec, dstr

OPEN, VAL, CLOSE = "<|num_start|>", "<|num_val|>", "<|num_end|>"
K_ADD, K_SUB, K_RSUB, K_MUL = 0, 1, 2, 3                     # u + y, u - y, y - u, u * y
NEG = -1e4                                                    # "impossible" in log space (finite: no NaN gradients)
_PREFIX = re.compile(r"(?:(?<![\w.])-)?\d+(?:\.\d*)?$")


def workspace(m):
    return "\n" + (OPEN + VAL + CLOSE) * m if m else ""


def num_prefix(text):
    """The beginning of a number that a text ends with ('' if it does not end inside a number)."""
    m = _PREFIX.search(text)
    if m:
        return m.group(0)
    if text.endswith("-") and (len(text) == 1 or text[-2] in " \n\t;:=(["):
        return "-"
    return ""


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def written(text):
    """The numbers that are already written completely in a text (the number it ends inside does not count)."""
    body = text[:len(text) - len(num_prefix(text))]
    return {canon(x) for x in _NUMBER.findall(body)} - {None}


def neg_string(s):
    s = str(s)
    return s[1:] if s.startswith("-") else ("0" if dec(s) == 0 else "-" + s)


def next_char(string, prefix):
    """The character with which `string` continues `prefix`, or None."""
    if string is not None and len(string) > len(prefix) and string.startswith(prefix):
        return string[len(prefix)]
    return None


class Layout:
    """Fixed slots of the registers of one prompt: the numbers 0..M-1, then for every operator layer l and number j
    the additive result (group 0) and the product (group 1). The order of the slots is the order of creation."""

    def __init__(self, max_spans, n_layers):
        self.M, self.L = max_spans, n_layers
        self.S = max_spans * (1 + 2 * n_layers)
        span, layer = list(range(max_spans)), [-1] * max_spans
        for l in range(n_layers):
            for j in range(max_spans):
                span += [j, j]
                layer += [l, l]
        self.span, self.layer = span, layer

    def slot(self, l, j, g):
        return self.M + (l * self.M + j) * 2 + g


def build_program(numbers, gold, lay, has_add=True, has_mul=True):
    """Distant supervision. numbers: the numbers of the prompt (strings); gold: the numbers of the target text.
    Returns {m, u, y, kind, string, cons}: for every result slot that produces a target number its operands, its
    operation and its (exact) string, and for every group (slot) that was offered a choice the list `cons` of all
    choices (u, y, kind) that produce a target number not produced before ([]: the right choice is "nothing")."""
    m = min(len(numbers), lay.M)
    val = {j: dec(numbers[j]) for j in range(m)}
    prog = {"m": m, "u": {}, "y": {}, "kind": {}, "string": {j: str(numbers[j]) for j in range(m)}, "cons": {}}
    remaining = {dec(g) for g in gold} - set(val.values())
    for l in range(lay.L):
        for j in range(1, m):
            first = lay.slot(l, j, 0)
            U = [s for s in sorted(val) if lay.span[s] < j and s < first]
            Y = [s for s in sorted(val) if lay.span[s] == j and lay.layer[s] < l]
            for g in (0, 1):
                if not (has_add, has_mul)[g]:
                    continue
                cons = []
                for u in U:
                    for y in Y:
                        cand = ((K_ADD, val[u] + val[y]), (K_SUB, val[u] - val[y]), (K_RSUB, val[y] - val[u])) if g == 0 \
                            else ((K_MUL, val[u] * val[y]),)
                        cons += [(u, y, k, v) for k, v in cand if v in remaining and dstr(v) is not None]
                s = lay.slot(l, j, g)
                prog["cons"][s] = [c[:3] for c in cons]
                if cons:
                    u, y, k, v = cons[0]
                    val[s] = v
                    prog["u"][s], prog["y"][s], prog["kind"][s], prog["string"][s] = u, y, k, dstr(v)
                    remaining.discard(v)
    return prog


def pack_programs(progs, lay):
    """The programs of a batch as tensors: for every result slot whether it exists and its operands and operation
    (teacher forcing), and for every slot that was offered a choice the padded list of the right choices of every row
    (rows, u, y, kind, mask) and the rows whose right choice is "nothing"."""
    B, M, S = len(progs), lay.M, lay.S
    pe = torch.zeros(B, S - M, dtype=torch.bool)
    pu, py, pk = (torch.zeros(B, S - M, dtype=torch.long) for _ in range(3))
    cons = {}
    for b, prog in enumerate(progs):
        for s in prog["u"]:
            pe[b, s - M], pu[b, s - M], py[b, s - M], pk[b, s - M] = True, prog["u"][s], prog["y"][s], prog["kind"][s]
        for s, cs in prog["cons"].items():
            cons.setdefault(s, []).append((b, cs))
    packed = {}
    for s, items in cons.items():
        some = [(b, cs) for b, cs in items if cs]
        width = max((len(cs) for _, cs in some), default=1)
        cu, cy, ck = (torch.zeros(len(some), width, dtype=torch.long) for _ in range(3))
        cm = torch.zeros(len(some), width, dtype=torch.bool)
        for r, (_, cs) in enumerate(some):
            for q, (u, y, k) in enumerate(cs):
                cu[r, q], cy[r, q], ck[r, q], cm[r, q] = u, y, min(k, 2), True     # the sign head has 3 classes
        packed[s] = (torch.tensor([b for b, _ in some], dtype=torch.long), cu, cy, ck, cm,
                     torch.tensor([b for b, cs in items if not cs], dtype=torch.long))
    return pe, pu, py, pk, packed


# ------------------------------------------------------------------------------------- numbers written in the answer
# Operations on the numbers that the model writes (step-by-step solutions). The a-th number of the answer gets three
# registers after the static ones: its value (slot S + 3a, encoded by the number model from the digits written), the
# result of one addition (S + 3a + 1) and of one multiplication (S + 3a + 2) of that number with an earlier register.
# The operand and the operation are chosen by pointers from the final hidden state of the number's last digit; the
# registers are available to the output mixture from the second token after the number (when the model has written
# the token that ends the number, at test time).
def build_dynamic(prog, ans, lay, A, window=3, has_add=True, has_mul=True):
    """Distant supervision for the numbers of the answer. prog: the static program (build_program); ans: the
    numbers of the target answer in order. A choice is right if its result is one of the next `window` numbers of
    the answer that is not yet the value of a register. Returns {n, u, kind, string, cons} like build_program."""
    S = lay.S
    val = {s: dec(v) for s, v in prog["string"].items() if v is not None}
    out = {"n": min(len(ans), A), "u": {}, "kind": {}, "string": {}, "cons": {}}
    for a in range(out["n"]):
        sv = S + 3 * a
        cur = dec(ans[a])
        avail = dict(val)                                   # the registers before this number
        val[sv] = cur
        out["string"][sv] = str(ans[a])
        targets = {dec(x) for x in ans[a + 1:a + 1 + window]} - set(val.values())
        for g in (0, 1):
            if not (has_add, has_mul)[g]:
                continue
            cons = []
            for u in sorted(avail):
                vu = avail[u]
                cand = ((K_ADD, vu + cur), (K_SUB, vu - cur), (K_RSUB, cur - vu)) if g == 0 else ((K_MUL, vu * cur),)
                cons += [(u, k, v) for k, v in cand if v in targets and dstr(v) is not None]
            s = sv + 1 + g
            out["cons"][s] = [(u, k) for u, k, _ in cons]
            if cons:
                u, k, v = cons[0]
                val[s] = v
                out["u"][s], out["kind"][s], out["string"][s] = u, k, dstr(v)
                targets.discard(v)
    return out


def pack_dynamic(dprogs, S, A):
    """The dynamic programs of a batch as tensors (teacher registers (B, A, 2) and the right choices per slot)."""
    B = len(dprogs)
    pe = torch.zeros(B, A, 2, dtype=torch.bool)
    pu, pk = torch.zeros(B, A, 2, dtype=torch.long), torch.zeros(B, A, 2, dtype=torch.long)
    cons = {}
    for b, dp in enumerate(dprogs):
        for s, u in dp["u"].items():
            a, g = (s - S) // 3, (s - S) % 3 - 1
            pe[b, a, g], pu[b, a, g], pk[b, a, g] = True, u, dp["kind"][s]
        for s, cs in dp["cons"].items():
            cons.setdefault(s, []).append((b, cs))
    packed = {}
    for s, items in cons.items():
        some = [(b, cs) for b, cs in items if cs]
        width = max((len(cs) for _, cs in some), default=1)
        cu, ck = torch.zeros(len(some), width, dtype=torch.long), torch.zeros(len(some), width, dtype=torch.long)
        cm = torch.zeros(len(some), width, dtype=torch.bool)
        for r, (_, cs) in enumerate(some):
            for q, (u, k) in enumerate(cs):
                cu[r, q], ck[r, q], cm[r, q] = u, min(k, 2), True
        packed[s] = (torch.tensor([b for b, _ in some], dtype=torch.long), cu, ck, cm,
                     torch.tensor([b for b, cs in items if not cs], dtype=torch.long))
    return pe, pu, pk, packed


def number_ended(prev, new):
    """The number that `prev` ends with if the text `new` (= prev + the next token) does not continue it, else None."""
    p = num_prefix(prev)
    if not p or p == "-":
        return None
    q = num_prefix(new)
    if len(q) > len(p) and q.startswith(p):
        return None
    return canon(p.rstrip("."))


# ------------------------------------------------------------------------------------- the number model
class NIFNumbers(nn.Module):
    """A trained NIF, frozen and always in evaluation mode (a submodule of the language model, so that it moves with
    .to(); its weights are not trained and not saved with the language model)."""

    def __init__(self, ckpt):
        super().__init__()
        import os
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from nif.model import NIF
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.nif = NIF(**st["cfg"])
        self.nif.load_state_dict(st["model"])
        self.nif.eval().requires_grad_(False)
        self.dim, self.c = self.nif.d, self.nif.c

    def train(self, mode=True):                    # the NIF stays frozen and in evaluation mode
        return super().train(False)

    @torch.no_grad()
    def encode(self, strings):
        return self.nif.encode(list(strings))


class NIFAlgebra:
    """The frozen NIF: encoder, operators + and x, decoder. -v is computed as v x phi(-1)."""
    name, has = "nif", (True, True)

    def __init__(self, numbers):
        self.module, self.dim = numbers, numbers.dim
        self._m1 = None

    def encode(self, strings):
        return self.module.encode(strings).float()

    @torch.no_grad()
    def add(self, a, b):
        return self.module.nif.add(a, b).float()

    @torch.no_grad()
    def mul(self, a, b):
        return self.module.nif.mul(a, b).float()

    def neg(self, a):
        if self._m1 is None or self._m1.device != a.device:
            self._m1 = self.encode(["-1"]).to(a.device)
        return self.mul(a, self._m1.expand(a.shape[0], -1))

    @torch.no_grad()
    def decode(self, z):
        return [canon(s) for s in self.module.nif.decode(z)]


# ------------------------------------------------------------------------------------- modules
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight


def unit(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


class KeyNet(nn.Module):
    """Keys of the registers. A number: from the state of its value token (which has seen the whole problem) and
    the state of its last digit in the text. A result: from the keys of its two operands, its operation and its
    operator layer, so that a key exists as soon as the result does."""

    def __init__(self, H, d, n_layers):
        super().__init__()
        self.p_ws, self.p_txt = nn.Linear(H, d), nn.Linear(H, d)
        self.e_num = nn.Parameter(torch.zeros(d))
        self.w_l, self.w_r = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.e_kind, self.e_layer = nn.Embedding(4, d), nn.Embedding(n_layers, d)

    def numbers(self, h_val, h_txt):
        return unit(self.p_ws(h_val) + self.p_txt(h_txt) + self.e_num)

    def result(self, ku, ky, kind, layer):
        return unit(self.w_l(ku) + self.w_r(ky) + self.e_kind(kind) + self.e_layer.weight[layer])


class OpStage(nn.Module):
    """The adapters of one operator layer."""

    def __init__(self, H, d, dim, n_layers):
        super().__init__()
        self.norm = RMSNorm(H)
        self.q = nn.ModuleList([nn.Linear(H, d), nn.Linear(H, d)])        # operand u: from <|num_start|>, from <|num_end|>
        self.qy = nn.ModuleList([nn.Linear(H, d), nn.Linear(H, d)])       # own value y
        self.sign = nn.Linear(H, 3)                                        # u + y, u - y, y - u
        # every query also reads the state of the number's last digit in the text, which has seen the words and
        # signs just before the number (the tag states come after the whole problem)
        self.q_txt = nn.ModuleList([nn.Linear(H, d, bias=False), nn.Linear(H, d, bias=False)])
        self.qy_txt = nn.ModuleList([nn.Linear(H, d, bias=False), nn.Linear(H, d, bias=False)])
        self.sign_txt = nn.Linear(H, 3, bias=False)
        self.keys = KeyNet(H, d, n_layers)
        self.null = nn.Parameter(torch.randn(2, d) * 0.02)                 # "no operation"
        self.wb = nn.ModuleList([nn.Linear(dim, H) for _ in range(3)])    # inverse adapters: sum, number, product
        self.gate = nn.Parameter(torch.zeros(3))                           # tanh gates, closed at the start


class CopyHead(nn.Module):
    def __init__(self, H, d, n_layers):
        super().__init__()
        self.norm, self.q = RMSNorm(H), nn.Linear(H, d)
        self.keys = KeyNet(H, d, n_layers)
        self.gen = nn.Parameter(torch.randn(d) * 0.02)
        self.gen_bias = nn.Parameter(torch.tensor(4.0))                    # at the start the language model writes
        self.cov = nn.Parameter(torch.zeros(()))                           # coverage: a register that is already written


class DynHead(nn.Module):
    """Pointers for the numbers written in the answer (read the final hidden state of the number's last digit)."""

    def __init__(self, H, d):
        super().__init__()
        self.norm = RMSNorm(H)
        self.q = nn.ModuleList([nn.Linear(H, d), nn.Linear(H, d)])        # operand of the addition, of the product
        self.sign = nn.Linear(H, 3)                                        # u + y, u - y, y - u
        self.null = nn.Parameter(torch.randn(2, d) * 0.02)
        self.tag = nn.Parameter(torch.zeros(d))                            # marks the keys of these results


class Ctx:
    """The registers of one batch during one forward pass."""


class OpModule(nn.Module):
    def __init__(self, H, dim, layers, max_spans=12, d=256, writeback=True, copy=True, dyn_max=0, dyn_window=3):
        super().__init__()
        self.lay = Layout(max_spans, len(layers))
        self.layers, self.d, self.dim, self.writeback, self.copy_on = list(layers), d, dim, writeback, copy
        self.A, self.window = int(dyn_max), int(dyn_window)                # numbers of the answer with registers
        self.stages = nn.ModuleList([OpStage(H, d, dim, len(layers)) for _ in layers])
        self.copy = CopyHead(H, d, len(layers))
        if self.A:
            self.dyn = DynHead(H, d)
        self.in_proj = nn.Linear(dim, H)                                   # input adapter of the value token
        self.in_gate = nn.Parameter(torch.zeros(()))
        self.tag_emb = nn.Parameter(torch.zeros(3, H))                     # <|num_start|>, <|num_val|>, <|num_end|>
        self.register_buffer("span_t", torch.tensor(self.lay.span), persistent=False)
        self.register_buffer("layer_t", torch.tensor(self.lay.layer), persistent=False)

    # --------------------------------------------------------------------------------- one batch
    def begin(self, ob, alg, device, teacher, shuffle=False):
        """ob: the batch description built by NumLLM._op_batch. Encodes the numbers and creates the registers."""
        lay, c = self.lay, Ctx()
        B = ob["B"]
        c.B, c.teacher, c.alg, c.max_m = B, teacher, alg, int(max(ob["m"]) if ob["m"] else 0)
        for k in ("open", "val", "close", "txt", "span_exists"):
            setattr(c, k, ob[k].to(device))
        c.bi, c.ji = c.span_exists.nonzero(as_tuple=True)                 # every number of the batch
        strings = [s for row in ob["numbers"] for s in row]
        c.V = torch.zeros(B, lay.S, self.dim, device=device)
        c.NegV = torch.zeros(B, lay.M, self.dim, device=device)
        c.num_strings = [list(row) for row in ob["numbers"]]
        if strings:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
                e = alg.encode(strings).float().to(device)
                en = alg.encode([s[1:] if s.startswith("-") else ("0" if dec(s) == 0 else "-" + s) for s in strings]).float().to(device)
            if shuffle:                                                    # test: the vectors of other numbers
                perm = torch.roll(torch.arange(len(strings), device=device), 1)
                e, en = e[perm], en[perm]
                dec_s = alg.decode(e)
                k = 0
                for row in c.num_strings:
                    for i in range(len(row)):
                        row[i] = dec_s[k]
                        k += 1
            c.V[c.bi, c.ji] = e
            c.NegV[c.bi, c.ji] = en
        c.exists = torch.zeros(B, lay.S, dtype=torch.bool, device=device)
        c.exists[:, :lay.M] = c.span_exists
        c.u = torch.zeros(B, lay.S, dtype=torch.long, device=device)
        c.y = torch.zeros(B, lay.S, dtype=torch.long, device=device)
        c.kind = torch.zeros(B, lay.S, dtype=torch.long, device=device)
        c.cons = {}
        if teacher:
            for k in ("exists", "u", "y", "kind"):
                getattr(c, k)[:, lay.M:] = ob["prog_" + k].to(device)
            c.cons = {s: tuple(t.to(device) for t in v) for s, v in ob["cons"].items()}
        c.losses, c.hits, c.deltas, c.phase = [], [], {}, "run"
        return c

    def _result_keys(self, keys, K, c, upto):
        lay = self.lay
        for l in range(upto):
            for j in range(1, c.max_m):
                for g in (0, 1):
                    s = lay.slot(l, j, g)
                    rows = c.exists[:, s].nonzero().squeeze(1)
                    if rows.numel():
                        K = K.index_put((rows, torch.full_like(rows, s)),
                                        keys.result(K[rows, c.u[rows, s]], K[rows, c.y[rows, s]], c.kind[rows, s], l))
        return K

    def _number_keys(self, keys, norm, h, c):
        K = torch.zeros(c.B, self.lay.S, self.d, device=h.device)
        if c.bi.numel():
            hv = norm(h[c.bi, c.val[c.bi, c.ji]].float())
            ht = norm(h[c.bi, c.txt[c.bi, c.ji]].float())
            K = K.index_put((c.bi, c.ji), keys.numbers(hv, ht))
        return K

    def _neg(self, c, rows, slots):
        out = c.V[rows, slots].clone()
        isnum = slots < self.lay.M
        if isnum.any():
            out[isnum] = c.NegV[rows[isnum], slots[isnum]]
        if (~isnum).any():
            out[~isnum] = c.alg.neg(c.V[rows[~isnum], slots[~isnum]])
        return out

    @torch.no_grad()
    def _operate(self, c, rows, u, y, kind):
        a, b = c.V[rows, u], c.V[rows, y]
        out = torch.zeros_like(a)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.is_cuda):
            for k in (K_ADD, K_SUB, K_RSUB, K_MUL):
                m = kind == k
                if not m.any():
                    continue
                if k == K_ADD:
                    out[m] = c.alg.add(a[m], b[m])
                elif k == K_SUB:
                    out[m] = c.alg.add(a[m], self._neg(c, rows[m], y[m]))
                elif k == K_RSUB:
                    out[m] = c.alg.add(b[m], self._neg(c, rows[m], u[m]))
                else:
                    out[m] = c.alg.mul(a[m], b[m])
        return out.float()

    def _pointer_loss(self, c, s, lpu, lpy, lps, uh, yh, kh):
        rows, cu, cy, ck, cm, null = c.cons[s]
        if rows.numel():
            r = rows[:, None]
            lp = lpu[r, cu] + lpy[r, cy] + (lps[r, ck] if lps is not None else 0.0)
            c.losses.append(-torch.logsumexp(lp.masked_fill(~cm, NEG), 1))
            hit = (uh[rows][:, None] == cu) & (yh[rows][:, None] == cy) & cm
            if lps is not None:
                hit = hit & (kh[rows][:, None] == ck)
            c.hits.append(hit.any(1).float())
        if null.numel():
            c.losses.append(-lpu[null, -1])
            c.hits.append((uh[null] == lpu.shape[1] - 1).float())

    def run_stage(self, i, h, c):
        """Operator layer i on the hidden states h (B, T, H) of the language model. Returns the new hidden states."""
        st, lay, d = self.stages[i], self.lay, self.d
        S, dev = lay.S, h.device
        if not c.bi.numel():
            return h
        with torch.autocast("cuda", enabled=False):
            K = self._result_keys(st.keys, self._number_keys(st.keys, st.norm, h, c), c, i)
            h_open = st.norm(h[c.bi, c.open[c.bi, c.ji]].float())
            h_close = st.norm(h[c.bi, c.close[c.bi, c.ji]].float())
            h_txt = st.norm(h[c.bi, c.txt[c.bi, c.ji]].float())
            dense = lambda x: torch.zeros(c.B, lay.M, x.shape[-1], device=dev).index_put((c.bi, c.ji), x)
            Q = [dense(st.q[0](h_open) + st.q_txt[0](h_txt)), dense(st.q[1](h_close) + st.q_txt[1](h_txt))]
            QY = [dense(st.qy[0](h_open) + st.qy_txt[0](h_txt)), dense(st.qy[1](h_close) + st.qy_txt[1](h_txt))]
            SG = dense(st.sign(h_open) + st.sign_txt(h_txt))
            slot_ids = torch.arange(S, device=dev)
            for j in range(1, c.max_m):
                act = c.span_exists[:, j]
                first = lay.slot(i, j, 0)
                mU = c.exists & ((self.span_t < j) & (slot_ids < first))[None]
                mY = c.exists & ((self.span_t == j) & (self.layer_t < i))[None]
                mY[:, 0] |= ~mY.any(1)                                     # rows without this number: a dummy entry
                for g in (0, 1):
                    if not c.alg.has[g]:
                        continue
                    s = lay.slot(i, j, g)
                    q, qy = Q[g][:, j], QY[g][:, j]
                    su = torch.einsum("bsd,bd->bs", K, q) / math.sqrt(d)
                    su = torch.cat([su.masked_fill(~mU, float("-inf")), (q @ st.null[g])[:, None] / math.sqrt(d)], 1)
                    sy = (torch.einsum("bsd,bd->bs", K, qy) / math.sqrt(d)).masked_fill(~mY, float("-inf"))
                    lpu, lpy = F.log_softmax(su, 1), F.log_softmax(sy, 1)
                    lps = F.log_softmax(SG[:, j], 1) if g == 0 else None
                    uh, yh = lpu.argmax(1), lpy.argmax(1)
                    kh = lps.argmax(1) if g == 0 else torch.full_like(uh, K_MUL)
                    if c.teacher:
                        if s in c.cons:
                            self._pointer_loss(c, s, lpu, lpy, lps, uh, yh, kh)
                    else:                                                  # test: every choice is the arg max
                        c.exists[:, s] = act & (uh < S)
                        c.u[:, s], c.y[:, s], c.kind[:, s] = uh.clamp(max=S - 1), yh, kh
                    rows = c.exists[:, s].nonzero().squeeze(1)
                    if rows.numel():
                        u, y, k = c.u[rows, s], c.y[rows, s], c.kind[rows, s]
                        K = K.index_put((rows, torch.full_like(rows, s)), st.keys.result(K[rows, u], K[rows, y], k, i))
                        c.V[rows, s] = self._operate(c, rows, u, y, k)
            return self._write_back(i, st, h, c)

    def _write_back(self, i, st, h, c):
        """Inverse adapters: sum -> <|num_start|>, number -> value token, product -> <|num_end|> (gated)."""
        if not self.writeback:
            return h
        lay = self.lay
        scale = h.detach().float().pow(2).mean().sqrt()
        js = torch.arange(lay.M, device=h.device)
        bs, ps, vs = [], [], []
        for k, (pos, g) in enumerate(((c.open, 0), (c.val, None), (c.close, 1))):
            if g is None:
                b, j = c.bi, c.ji
                vec = c.V[b, j]
            else:
                sl = lay.slot(i, 0, g) + 2 * js                            # the slots of this layer and group
                b, j = c.exists[:, sl].nonzero(as_tuple=True)
                vec = c.V[b, sl[j]]
            if b.numel():
                v = unit(st.wb[k](vec)) * (scale * torch.tanh(st.gate[k]))
                bs.append(b), ps.append(pos[b, j]), vs.append(v)
        if not bs:
            return h
        b, p, v = torch.cat(bs), torch.cat(ps), torch.cat(vs)
        c.deltas[i] = (b, p, v.detach())
        return h.index_put((b, p), v.to(h.dtype), accumulate=True)

    def replay(self, i, h, c):
        """Decoding without a cache: the prompt is recomputed, the stored vectors are written again."""
        if i not in c.deltas:
            return h
        b, p, v = c.deltas[i]
        return h.index_put((b, p), v.to(h.dtype), accumulate=True)

    def pointer_loss(self, c):
        if not c.losses:
            return None, None
        return torch.cat(c.losses).mean(), torch.cat(c.hits).mean()

    # --------------------------------------------------------------------------------- numbers written in the answer
    @property
    def S_total(self):
        return self.lay.S + 3 * self.A

    @property
    def S_alloc(self):
        """Slots of a batch at test time."""
        return self.S_total

    def _dyn_alloc(self, c, Kc):
        """Registers of every slot (static ones and those of the numbers of the answer) for one batch."""
        B, S, M, A, dev = c.B, self.lay.S, self.lay.M, self.A, Kc.device
        extra = self.S_alloc - S if not c.teacher else 3 * A
        c.K = torch.cat([Kc, Kc.new_zeros(B, extra, self.d)], 1)
        c.Vd = torch.cat([c.V, c.V.new_zeros(B, extra, self.dim)], 1)
        c.Negd = c.V.new_zeros(B, S + extra, self.dim)
        c.Negd[:, :M] = c.NegV
        c.isneg = torch.zeros(B, S + extra, dtype=torch.bool, device=dev)
        c.isneg[:, :M] = c.span_exists
        c.ex = torch.cat([c.exists, torch.zeros(B, extra, dtype=torch.bool, device=dev)], 1)
        c.dlosses, c.dhits = [], []

    def _dyn_encode(self, c, rows, slots, strings):
        """The numbers written in the answer, encoded from their digits (and their negatives, for u - y)."""
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=c.Vd.is_cuda):
            e = c.alg.encode(strings).float().to(c.Vd.device)
            en = c.alg.encode([neg_string(s) for s in strings]).float().to(c.Vd.device)
        c.Vd[rows, slots] = e
        c.Negd[rows, slots] = en
        c.isneg[rows, slots] = True
        c.ex[rows, slots] = True

    @torch.no_grad()
    def _dyn_combine(self, c, rows, u, y, kind):
        a, b = c.Vd[rows, u], c.Vd[rows, y]
        out = torch.zeros_like(a)

        def neg(slots, rr):
            o = c.Vd[rr, slots].clone()
            m = c.isneg[rr, slots]
            if m.any():
                o[m] = c.Negd[rr[m], slots[m]]
            if (~m).any():
                o[~m] = c.alg.neg(c.Vd[rr[~m], slots[~m]])
            return o
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.is_cuda):
            for k in (K_ADD, K_SUB, K_RSUB, K_MUL):
                m = kind == k
                if not m.any():
                    continue
                if k == K_ADD:
                    out[m] = c.alg.add(a[m], b[m])
                elif k == K_SUB:
                    out[m] = c.alg.add(a[m], neg(y[m], rows[m]))
                elif k == K_RSUB:
                    out[m] = c.alg.add(b[m], neg(u[m], rows[m]))
                else:
                    out[m] = c.alg.mul(a[m], b[m])
        return out.float()

    def _dyn_scores(self, K, ex, g, grp, sv):
        """log-probabilities over the registers before slot sv (per row) and "nothing" (last column)."""
        q = self.dyn.q[grp](g)
        sc = torch.einsum("nsd,nd->ns", K, q) / math.sqrt(self.d)
        slot = torch.arange(K.shape[1], device=K.device)
        ok = ex & (slot[None] < sv[:, None])
        sc = torch.cat([sc.masked_fill(~ok, float("-inf")), (q @ self.dyn.null[grp])[:, None] / math.sqrt(self.d)], 1)
        return F.log_softmax(sc, 1)

    def _dyn_loss(self, c, cons, lpu, lps, uh, kh):
        rows, cu, ck, cm, null = cons
        if rows.numel():
            r = rows[:, None]
            lp = lpu[r, cu] + (lps[r, ck] if lps is not None else 0.0)
            lp = torch.nan_to_num(lp, neginf=NEG).masked_fill(~cm, NEG)
            c.dlosses.append(-torch.logsumexp(lp, 1))
            hit = (uh[rows][:, None] == cu) & cm
            if lps is not None:
                hit = hit & (kh[rows][:, None] == ck)
            c.dhits.append(hit.any(1).float())
        if null.numel():
            c.dlosses.append(-lpu[null, -1])
            c.dhits.append((uh[null] == lpu.shape[1] - 1).float())

    def dyn_loss(self, c):
        if not getattr(c, "dlosses", None):
            return None, None
        return torch.cat(c.dlosses).mean(), torch.cat(c.dhits).mean()

    def dyn_train(self, h, c, dob, Kc):
        """Teacher-forced operations on the numbers of the target answer. h: final hidden states (B, T, H); dob: the
        numbers of the answer, their positions and programs (NumLLM._op_batch); Kc: keys of the static registers.
        Returns the keys of all registers (B, S + 3A, d)."""
        if not self.A:
            return Kc
        with torch.autocast("cuda", enabled=False):
            self._dyn_alloc(c, Kc)
            B, S, dev = c.B, self.lay.S, h.device
            ex_ans, pos = dob["exists"].to(dev), dob["pos"].to(dev)
            rows_all, a_all = ex_ans.nonzero(as_tuple=True)
            if not rows_all.numel():
                return c.K
            strings = [dob["strings"][b][a] for b, a in zip(rows_all.tolist(), a_all.tolist())]
            self._dyn_encode(c, rows_all, S + 3 * a_all, strings)
            pe, pu, pk = dob["prog_exists"].to(dev), dob["prog_u"].to(dev), dob["prog_kind"].to(dev)
            cons = {s: tuple(t.to(dev) for t in v) for s, v in dob["cons"].items()}
            ar = torch.arange(B, device=dev)
            for a in range(int(ex_ans.sum(1).max())):
                rows = ex_ans[:, a].nonzero().squeeze(1)
                if not rows.numel():
                    continue
                sv = S + 3 * a
                g = self.dyn.norm(h[ar, pos[:, a]].float())                  # every row; rows without number a unused
                c.K = c.K.index_put((rows, torch.full_like(rows, sv)), self.copy.keys.numbers(g[rows], g[rows]))
                svt = torch.full((B,), sv, device=dev)
                for grp in (0, 1):
                    if not c.alg.has[grp]:
                        continue
                    s = sv + 1 + grp
                    lpu = self._dyn_scores(c.K, c.ex, g, grp, svt)
                    lps = F.log_softmax(self.dyn.sign(g), 1) if grp == 0 else None
                    uh = lpu.argmax(1)
                    kh = lps.argmax(1) if lps is not None else None
                    if s in cons:
                        self._dyn_loss(c, cons[s], lpu, lps, uh, kh)
                    tr = pe[:, a, grp].nonzero().squeeze(1)
                    if tr.numel():
                        u, k = pu[tr, a, grp], pk[tr, a, grp]
                        y = torch.full_like(tr, sv)
                        c.Vd[tr, s] = self._dyn_combine(c, tr, u, y, k)
                        kr = self.copy.keys.result(c.K[tr, u], c.K[tr, y], k, 0) + self.dyn.tag
                        c.K = c.K.index_put((tr, torch.full_like(tr, s)), kr)
                        c.ex[tr, s] = True
        return c.K

    def dyn_begin(self, c, Kc):
        """Generation: the registers of every slot; the numbers of the answer are added by dyn_step."""
        if not self.A:
            c.K = Kc
            return
        with torch.autocast("cuda", enabled=False):
            self._dyn_alloc(c, Kc)
        c.n_ans = [0] * c.B

    @torch.no_grad()
    def dyn_step(self, c, rows, values, g):
        """Generation: rows whose number has just ended (values: its canonical strings; g (B, H): the final hidden
        states of the last input tokens, i.e. of the numbers' last digits for these rows). Encodes the numbers,
        applies the chosen operations and returns {(row, slot): string} of the new registers (the numbers and their
        results, decoded by the number model)."""
        out = {}
        keep = [(i, b) for i, b in enumerate(rows) if values[i] is not None and c.n_ans[b] < self.A]
        if not self.A or not keep:
            return out
        dev = g.device
        rr = torch.tensor([b for _, b in keep], device=dev)
        idx = rr
        sv = self.lay.S + 3 * torch.tensor([c.n_ans[b] for _, b in keep], device=dev)
        for _, b in keep:
            c.n_ans[b] += 1
        strs = [values[i] for i, _ in keep]
        with torch.autocast("cuda", enabled=False):
            self._dyn_encode(c, rr, sv, strs)
            gg = self.dyn.norm(g[idx].float())
            c.K[rr, sv] = self.copy.keys.numbers(gg, gg)
            for b, s, st in zip(rr.tolist(), sv.tolist(), strs):
                out[(b, s)] = st
            for grp in (0, 1):
                if not c.alg.has[grp]:
                    continue
                lpu = self._dyn_scores(c.K[rr], c.ex[rr], gg, grp, sv)
                uh = lpu.argmax(1)
                kh = self.dyn.sign(gg).argmax(1) if grp == 0 else torch.full_like(uh, K_MUL)
                sel = uh < lpu.shape[1] - 1
                if not sel.any():
                    continue
                r2, u, k, sv2 = rr[sel], uh[sel], kh[sel], sv[sel]
                s = sv2 + 1 + grp
                c.Vd[r2, s] = self._dyn_combine(c, r2, u, sv2, k)
                c.K[r2, s] = self.copy.keys.result(c.K[r2, u], c.K[r2, sv2], k, 0) + self.dyn.tag
                c.ex[r2, s] = True
                for b, ss, st in zip(r2.tolist(), s.tolist(), c.alg.decode(c.Vd[r2, s])):
                    if st is not None:
                        out[(b, ss)] = st
        return out

    # --------------------------------------------------------------------------------- input and output
    def embed_tags(self, x, ids, tag_ids, c, emb_rms):
        """Embeddings of the workspace tokens: trained tag embeddings; value token + input adapter of the vector."""
        x = x.clone()
        for k, t in enumerate(tag_ids):
            m = ids == t
            if m.any():
                x[m] = self.tag_emb[k].to(x.dtype)
        if c is not None and c.bi.numel():
            v = unit(self.in_proj(c.V[c.bi, c.ji])) * (emb_rms * torch.tanh(self.in_gate))
            x = x.index_put((c.bi, c.val[c.bi, c.ji]), v.to(x.dtype), accumulate=True)
        return x

    def copy_keys(self, h, c):
        with torch.autocast("cuda", enabled=False):
            return self._result_keys(self.copy.keys, self._number_keys(self.copy.keys, self.copy.norm, h, c), c, self.lay.L)

    def _pointer(self, hs, Kc, nt, used):
        """log-probabilities over [every register, the language model] at n positions. nt (n, S): id of the next
        token of each register at the position, -1 if the register does not continue what is being written; used
        (n, S): the register's number is already written in the answer (coverage, as in pointer-generator networks)."""
        q = self.copy.q(self.copy.norm(hs.float()))
        sc = torch.einsum("nsd,nd->ns", Kc, q) / math.sqrt(self.d) + self.copy.cov * used.float()
        gen = (q @ self.copy.gen) / math.sqrt(self.d) + self.copy.gen_bias
        return F.log_softmax(torch.cat([sc.masked_fill(nt < 0, float("-inf")), gen[:, None]], 1), 1)

    def mix_loss(self, hs, logits, tgt, b_sel, nt, used, Kc):
        """Negative log-likelihood of the mixture at the supervised positions, and statistics. stats["lm_loss"]: the
        negative log-likelihood of the language model alone (with gradient), for the auxiliary loss w_lm."""
        lm = -F.cross_entropy(logits.float(), tgt, reduction="none")
        if not self.copy_on:
            return -lm.mean(), {"lm": -lm.mean().detach(), "lm_loss": -lm.mean()}
        with torch.autocast("cuda", enabled=False):
            lp = self._pointer(hs, Kc[b_sel], nt, used)
            match = nt == tgt[:, None]
            lc = torch.logsumexp(lp[:, :-1].masked_fill(~match, NEG), 1)
            ll = torch.logaddexp(lp[:, -1] + lm, lc)
            has = match.any(1)
            stats = {"lm": -lm.mean().detach(), "copy_cov": has.float().mean().detach(), "lm_loss": -lm.mean()}
            if has.any():
                stats["copy_p"] = (lc[has] - ll[has]).exp().mean().detach()   # share of the mixture that is copied
            return -ll.mean(), stats

    def mix_next(self, hl, logits, Kc, nt, used):
        """Greedy next token of the mixture. Returns (ids, True where the copied part decided)."""
        p = F.softmax(logits.float(), -1)
        if not self.copy_on or Kc is None:
            return p.argmax(-1), torch.zeros(p.shape[0], dtype=torch.bool, device=p.device)
        with torch.autocast("cuda", enabled=False):
            lp = self._pointer(hl, Kc, nt, used)
            mix = p * lp[:, -1:].exp()
            lm_best = mix.max(-1).values
            mix = mix.scatter_add(1, nt.clamp(min=0), (lp[:, :-1].exp() * (nt >= 0)).to(mix.dtype))
            nxt = mix.argmax(-1)
            return nxt, mix.gather(1, nxt[:, None]).squeeze(1) > lm_best * 1.0000001
