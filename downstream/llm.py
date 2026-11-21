"""The language model for the experiment of Section 5.4: Qwen3.5-0.8B with numbers written as digits, optionally with
NIF operator layers (oplayers.py).

Speed: the output layer (248k words) is applied only where a token is predicted (the supervised positions in
training, the last position in decoding); on the GPU the frozen model is kept in bfloat16 (the precision of the
checkpoint); the linear-attention layers use the Triton kernels of flash-linear-attention when it is installed
(NIF_DS_KERNELS=torch forces the reference PyTorch path, for comparisons).
"""
from __future__ import annotations

import math
import os
import sys

if os.environ.get("NIF_DS_KERNELS", "fla") == "torch":          # must happen before transformers is imported
    for _m in ("fla", "causal_conv1d"):
        sys.modules.setdefault(_m, None)                         # makes `import fla` fail: PyTorch fallback

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from data import Num, render  # noqa: E402

INTERFACE = ("op.",)                                             # trained parameters besides the language model


def _uses_kernel(fn):
    """True if a function of the transformers model code was bound to an optimized package, False if it runs
    its PyTorch reference implementation, None if this cannot be read."""
    for _ in range(4):
        code, cl = getattr(fn, "__code__", None), getattr(fn, "__closure__", None)
        if code is not None and cl:
            cells = dict(zip(code.co_freevars, cl))
            if "is_new_implementation" in cells:
                return bool(cells["is_new_implementation"].cell_contents)
        fn = getattr(fn, "__wrapped__", None)
        if fn is None:
            break
    return None


def kernel_report():
    """Which implementation the linear-attention layers of Qwen3.5 use."""
    out = {"requested": os.environ.get("NIF_DS_KERNELS", "fla")}
    try:
        import fla
        out["fla"] = getattr(fla, "__version__", "installed")
    except Exception:
        out["fla"] = None
    try:
        import importlib
        m = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
        for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "fused_recurrent_gated_delta_rule",
                     "causal_conv1d_fn", "causal_conv1d_update"):
            if hasattr(m, name):
                out[name] = _uses_kernel(getattr(m, name))
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:100]}"
    return out


def load_lm(name, dtype=torch.float32):
    """The text decoder of `name`. Qwen3.5 checkpoints are vision-language models; they are loaded with their
    own class and used with text only."""
    import transformers as T
    errors = []
    for cls in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModelForMultimodalLM"):
        if not hasattr(T, cls):
            continue
        try:
            model, info = getattr(T, cls).from_pretrained(name, dtype=dtype, output_loading_info=True)
        except Exception as e:                            # this class cannot load the checkpoint
            errors.append(f"{cls}: {type(e).__name__}: {str(e)[:160]}")
            continue
        missing = [k for k in info["missing_keys"] if "lm_head" not in k]
        if missing:                                       # weights left at their random initialisation
            errors.append(f"{cls}: {len(missing)} weights are not in the checkpoint, e.g. {missing[0]}")
            continue
        return model, cls
    raise RuntimeError(f"could not load {name}: " + " | ".join(errors))


class NumLLM(nn.Module):
    """A causal language model; numbers are written as digits."""

    def __init__(self, name, dtype=torch.float32, ops=None, op_cfg=None, full_ft=False):
        """ops: the number model with operators (oplayers.NIFAlgebra) for operator layers: the numbers stay digit
        tokens, a workspace of tagged value tokens follows the prompt, the frozen operators run inside chosen layers
        and the answer is written by a pointer-generator mixture (see oplayers.py).
        full_ft: every weight of the language model is trained (load it in float32, the master weights)."""
        super().__init__()
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model, self.loader = load_lm(name, dtype)
        self.full_ft = bool(full_ft)
        self.model.requires_grad_(self.full_ft)
        emb = self.model.get_input_embeddings().weight
        self.H = emb.shape[1]
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else (self.tok.eos_token_id or 0)
        self.stop_ids = {i for i in (self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>"),
                                     self.tok.convert_tokens_to_ids("<|endoftext|>"))
                         if isinstance(i, int) and i >= 0 and i != self.tok.unk_token_id}
        self.eos_id = self.tok.convert_tokens_to_ids("<|im_end|>")
        if not isinstance(self.eos_id, int) or self.eos_id < 0 or self.eos_id == self.tok.unk_token_id:
            self.eos_id = self.tok.eos_token_id
        self.use_cache = True
        self.compiled = None                                              # torch.compile of the decoder (optional)
        self._wrap = self._template_parts()
        self.alg, self.op = ops, None
        if ops is not None:
            assert self._wrap is not None, "operator layers need the chat template split into prefix and suffix"
            from oplayers import CLOSE, OPEN, VAL, OpModule
            self.tok.add_special_tokens({"additional_special_tokens": [OPEN, VAL, CLOSE]})
            self.tag_ids = [self.tok.convert_tokens_to_ids(t) for t in (OPEN, VAL, CLOSE)]   # never looked up in the table
            self.opv = ops.module                                          # the frozen number model (moves with .to())
            self.op = OpModule(self.H, ops.dim, op_cfg["layers"], op_cfg["max_spans"], op_cfg["d_key"],
                               op_cfg.get("writeback", True), op_cfg.get("copy", True),
                               op_cfg.get("dyn_max", 0), op_cfg.get("dyn_window", 3))
            with torch.no_grad():                                          # new tokens start at the mean embedding
                e = emb.detach().float()
                self.op.tag_emb.copy_(e.mean(0)[None] + 0.02 * e.std() * torch.randn(3, self.H))
            self.register_buffer("emb_rms", emb.detach().float().pow(2).mean().sqrt(), persistent=False)
            self.c2id = {}                                                  # characters of a number that are one token
            for ch in "0123456789.-":
                t = self.tok(ch, add_special_tokens=False)["input_ids"]
                if len(t) == 1:
                    self.c2id[ch] = t[0]
            self._tokstr = {}
            self.w_ptr = 1.0                                                # weight of the pointer loss
            self.w_lm = 0.0                                                 # weight of the language model's own loss
            self.op_mode = "on"                                            # "off": no operator layers; "shuffle": wrong vectors
            self.op_log = {"prompts": 0, "numbers": 0, "registers": 0, "tokens": 0, "copied": 0}
            self._ctx = None
            n_layers = len(self.decoder().layers)
            assert all(0 <= l < n_layers for l in op_cfg["layers"]), f"operator layers {op_cfg['layers']} of {n_layers}"
            self._op_hooks = [self.decoder().layers[l].register_forward_hook(
                lambda m, a, o, i=i: self._op_hook(i, o)) for i, l in enumerate(op_cfg["layers"])]

    # ------------------------------------------------------------------------------- parameters
    def interface_parameters(self):
        return [p for n, p in self.named_parameters() if n.startswith(INTERFACE)]

    def base_parameters(self):
        """The weights of the language model (trained only with full fine-tuning)."""
        return [p for n, p in self.named_parameters() if n.startswith("model.")]

    def _trained_name(self, n):
        return n.startswith(INTERFACE) or (self.full_ft and n.startswith("model."))

    def set_trainable(self):
        """The operator layers (and every weight of the language model with full fine-tuning) are trained."""
        if self.full_ft:
            for p in self.base_parameters():
                p.requires_grad_(True)
        for p in self.interface_parameters():
            p.requires_grad_(True)

    def trainable_state(self):
        return {n: p.detach().cpu() for n, p in self.named_parameters() if self._trained_name(n)}

    def load_trainable(self, state):
        own = dict(self.named_parameters())
        missing = [n for n in own if self._trained_name(n) and n not in state]
        if missing:
            raise RuntimeError(f"checkpoint lacks {len(missing)} trained tensors, e.g. {missing[:3]}")
        with torch.no_grad():
            for n, v in state.items():
                own[n].copy_(v.to(own[n].device, own[n].dtype))

    # ------------------------------------------------------------------------------- text <-> tokens
    def _chat(self, user_text):
        msgs = [{"role": "user", "content": user_text}]
        if getattr(self.tok, "chat_template", None):
            try:
                return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return "User: " + user_text + "\nAssistant: "

    def _template_parts(self):
        """(prefix, suffix, trim) of the chat template around the user text, so that the template is rendered once
        and not for every example (the Qwen template strips the text: trim); None (render every time) unless it
        reproduces the template exactly on the probes."""
        mark = "⁣NIFTEXT⁣"
        full = self._chat(mark)
        if full.count(mark) != 1:
            return None
        pre, suf = full.split(mark)
        probes = ["Compute: 12.5 + 7\nAnswer with the number only.", " leading space", "trailing newline\n",
                  "Which number is larger?\nA: 3\nB: 4\nAnswer with A or B.", "<num> and <num>", "été 50%"]
        for trim in (False, True):
            if all(pre + (t.strip() if trim else t) + suf == self._chat(t) for t in probes):
                return pre, suf, trim
        return None

    def prompt_text(self, user_text):
        if not self._wrap:
            return self._chat(user_text)
        pre, suf, trim = self._wrap
        return pre + (user_text.strip() if trim else user_text) + suf

    def prompt_ids(self, user_text):
        return self.tok(self.prompt_text(user_text), add_special_tokens=False)["input_ids"]

    def encode_examples(self, examples, with_answer=True):
        """-> for every example: prompt ids, answer ids (ending with the end token) and two lists that only operator
        layers fill (see _encode_op). One tokenizer call for the whole list."""
        if self.op is not None:
            return self._encode_op(examples, with_answer)
        texts = [self.prompt_text(render(e["user"])[0]) for e in examples]
        if with_answer:
            texts += [render(e["asst"])[0] for e in examples]
        n = len(examples)
        ids = self.tok(texts, add_special_tokens=False)["input_ids"]
        if not with_answer:
            return [(ids[i], [], [], []) for i in range(n)]
        return [(ids[i], ids[n + i] + [self.eos_id], [], []) for i in range(n)]

    def encode_example(self, ex, with_answer=True):
        return self.encode_examples([ex], with_answer)[0]

    # ------------------------------------------------------------------------------- operator layers: tokens
    def _encode_op(self, examples, with_answer):
        """Rows for operator layers: (prompt ids, answer ids, numbers of the prompt with the positions of their
        last digit in the text and of their three workspace tokens, numbers of the answer text). The prompt is the
        text with digits, followed by the workspace."""
        from oplayers import workspace
        M, n = self.op.lay.M, len(examples)
        nums = [[p for p in e["user"] if isinstance(p, Num)][:M] for e in examples]
        texts = [self.prompt_text(render(e["user"])[0] + workspace(len(k))) for e, k in zip(examples, nums)]
        if with_answer:
            texts += [render(e["asst"])[0] for e in examples]
        enc = self.tok(texts, add_special_tokens=False, return_offsets_mapping=True)
        rows = []
        for i, e in enumerate(examples):
            ids, m = enc["input_ids"][i], len(nums[i])
            tags = [[t for t, x in enumerate(ids) if x == tid] for tid in self.tag_ids]
            if any(len(p) != m for p in tags):
                raise RuntimeError("the workspace tokens were not tokenized as single tokens")
            found = self._number_positions_all(e["user"], enc["offset_mapping"][i])
            uv = {"numbers": [str(x) for x in nums[i]], "txt": [found.get(k, tags[1][k]) for k in range(m)],
                  "open": tags[0], "val": tags[1], "close": tags[2]}
            if with_answer:
                uv["ans_pos"] = self._answer_positions(e["asst"], enc["offset_mapping"][n + i])
                rows.append((ids, enc["input_ids"][n + i] + [self.eos_id], uv, [str(p) for p in e["asst"] if isinstance(p, Num)]))
            else:
                rows.append((ids, [], uv, []))
        return rows

    @staticmethod
    def _answer_positions(parts, offsets):
        """Token index (in the answer) of the last digit of every number of the answer, -1 if it was not found."""
        out, pos, j = [], 0, 0
        for p in parts:
            s = str(p)
            if isinstance(p, Num):
                last = pos + len(s) - 1
                while j + 1 < len(offsets) and offsets[j + 1][0] <= last:
                    j += 1
                out.append(j if offsets and offsets[j][0] <= last < offsets[j][1] else -1)
            pos += len(s)
        return out

    def _number_positions_all(self, parts, offsets):
        """{index of the number in the prompt: token index of its last digit}."""
        pre, _, trim = self._wrap
        user = "".join(str(p) for p in parts)
        shift = len(pre) - ((len(user) - len(user.lstrip())) if trim else 0)
        out, pos, j, k = {}, 0, 0, 0
        for p in parts:
            s = str(p)
            if isinstance(p, Num):
                last = shift + pos + len(s) - 1
                while j + 1 < len(offsets) and offsets[j + 1][0] <= last:
                    j += 1
                if offsets[j][0] <= last < offsets[j][1]:
                    out[k] = j
                k += 1
            pos += len(s)
        return out

    def _tok_string(self, t):
        s = self._tokstr.get(t)
        if s is None:
            s = self._tokstr[t] = "" if t in self.stop_ids else self.tok.decode([t])
        return s

    def _op_batch(self, rows, shifts, T, teacher):
        """The workspace positions of a batch (shifted by the padding). With `teacher`: the program of every example
        (distant supervision, oplayers.build_program) and, for every answer position, the next token of every
        register that continues the number being written there."""
        from oplayers import build_dynamic, build_program, next_char, num_prefix, pack_dynamic, pack_programs, written
        lay, A = self.op.lay, self.op.A
        B, M, S = len(rows), lay.M, lay.S + 3 * self.op.A
        pos = {k: torch.zeros(B, M, dtype=torch.long) for k in ("open", "val", "close", "txt")}
        ex = torch.zeros(B, M, dtype=torch.bool)
        ob = {"B": B, "m": [], "numbers": []}
        for b, (p, a, uv, gold) in enumerate(rows):
            m = len(uv["numbers"])
            ob["m"].append(m)
            ob["numbers"].append(uv["numbers"])
            if m:
                for k in pos:
                    pos[k][b, :m] = torch.tensor(uv[k], dtype=torch.long) + shifts[b]
                ex[b, :m] = True
        ob.update(pos, span_exists=ex)
        if not teacher:
            return ob
        nt = torch.full((B, T, S), -1, dtype=torch.long)
        used = torch.zeros(B, T, S, dtype=torch.bool)                     # coverage: the number is already written
        progs, dprogs = [], []
        dob = {"exists": torch.zeros(B, max(A, 1), dtype=torch.bool), "pos": torch.zeros(B, max(A, 1), dtype=torch.long),
               "strings": [[] for _ in range(B)]}
        for b, (p, a, uv, gold) in enumerate(rows):
            prog = build_program(uv["numbers"], gold, lay, *self.alg.has)
            progs.append(prog)
            regs = [(s, string, 0) for s, string in prog["string"].items()]   # (slot, string, first answer token)
            if A:
                found = [(g, q) for g, q in zip(gold, uv.get("ans_pos", [])) if q >= 0][:A]
                dp = build_dynamic(prog, [g for g, _ in found], lay, A, self.op.window, *self.alg.has)
                dprogs.append(dp)
                for ai, (g, q) in enumerate(found):
                    dob["exists"][b, ai], dob["pos"][b, ai] = True, len(p) + q
                    dob["strings"][b].append(g)
                    for s in (lay.S + 3 * ai, lay.S + 3 * ai + 1, lay.S + 3 * ai + 2):
                        if s in dp["string"]:                     # usable from the second token after the number
                            regs.append((s, dp["string"][s], q + 2))
            text = ""
            for k, t in enumerate(a):
                pre, seen = num_prefix(text), written(text)
                for s, string, k0 in regs:
                    if k < k0:
                        continue
                    ch = next_char(string, pre)
                    if ch in self.c2id:
                        nt[b, len(p) + k, s] = self.c2id[ch]
                    if string in seen:
                        used[b, len(p) + k, s] = True
                text += self._tok_string(t)
        pe, pu, py, pk, packed = pack_programs(progs, lay)
        ob.update(prog_exists=pe, prog_u=pu, prog_y=py, prog_kind=pk, nt=nt, used=used, cons=packed, programs=progs)
        if A:
            dpe, dpu, dpk, dpacked = pack_dynamic(dprogs, lay.S, A)
            dob.update(prog_exists=dpe, prog_u=dpu, prog_kind=dpk, cons=dpacked, programs=dprogs)
            ob["dyn"] = dob
        return ob

    def _op_hook(self, i, output):
        c = self._ctx
        if c is None or c.phase == "skip":
            return output
        h = output[0] if isinstance(output, tuple) else output
        h2 = self.op.replay(i, h, c) if c.phase == "replay" else self.op.run_stage(i, h, c)
        return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2

    def _register_strings(self, c):
        """{slot: string} of every register of every prompt: the numbers of the prompt, and the results as the
        decoder of the number model writes them (no result string is computed in any other way)."""
        M = self.op.lay.M
        out = [dict(enumerate(row)) for row in c.num_strings]
        b, s = c.exists[:, M:].nonzero(as_tuple=True)
        if b.numel():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=b.is_cuda):
                strs = c.alg.decode(c.V[b, s + M])
            for bb, ss, st in zip(b.tolist(), s.tolist(), strs):
                if st is not None:
                    out[bb][ss + M] = st
        for row in out:
            for k in [k for k, v in row.items() if v is None]:
                del row[k]
        self.op_log["prompts"] += c.B
        self.op_log["numbers"] += int(c.span_exists.sum())
        self.op_log["registers"] += int(b.numel())
        return out

    def collate(self, examples, device):
        rows = self.encode_examples(examples)
        T = max(len(p) + len(a) for p, a, _, _ in rows)
        ids = torch.full((len(rows), T), self.pad_id, dtype=torch.long)
        attn = torch.zeros(len(rows), T, dtype=torch.long)
        labels = torch.full((len(rows), T), -100, dtype=torch.long)
        for i, (p, a, uv, av) in enumerate(rows):
            n = len(p) + len(a)
            ids[i, :n] = torch.tensor(p + a)
            attn[i, :n] = 1
            labels[i, len(p):n] = torch.tensor(a)
        nb = device != "cpu" and torch.cuda.is_available()
        out = dict(ids=ids.to(device, non_blocking=nb), attn=attn.to(device, non_blocking=nb),
                   labels=labels.to(device, non_blocking=nb), tokens=int(attn.sum()), padded=len(rows) * T)
        if self.op is not None:
            out["op"] = self._op_batch(rows, [0] * len(rows), T, teacher=True)
        return out

    # ------------------------------------------------------------------------------- model
    def base(self):
        return self.model

    def decoder(self):
        """The stack of layers and the final norm, without the output layer."""
        return self.base().get_decoder()

    def lm_head(self, h):
        W = self.base().get_output_embeddings()
        return W(h.to(W.weight.dtype))

    def embed(self, ids, ctx=None):
        E = self.model.get_input_embeddings()
        if self.op is not None:                                # workspace tokens: never looked up in the table
            m = (ids == self.tag_ids[0]) | (ids == self.tag_ids[1]) | (ids == self.tag_ids[2])
            return self.op.embed_tags(E(ids.masked_fill(m, self.pad_id)), ids, self.tag_ids, ctx, self.emb_rms)
        return E(ids)

    def compile(self, log=print):
        """torch.compile of the decoder for the training forward pass (shapes vary: dynamic). Optional."""
        try:
            self.compiled = torch.compile(self.decoder(), dynamic=True)
            log("decoder compiled with torch.compile (dynamic shapes) for training")
        except Exception as e:
            self.compiled = None
            log(f"torch.compile not used: {type(e).__name__}: {str(e)[:160]}")

    def hidden(self, x, attn, past=None, cache=False, train=False):
        """Final hidden states (after the last norm) and the cache."""
        dec = self.compiled if (train and self.compiled is not None) else self.decoder()
        try:
            out = dec(inputs_embeds=x, attention_mask=attn, past_key_values=past, use_cache=cache, return_dict=True)
        except Exception as e:
            if dec is self.compiled and self.compiled is not None:          # a compile failure: back to eager mode
                print(f"torch.compile failed ({type(e).__name__}: {str(e)[:160]}); eager mode from now on", flush=True)
                self.compiled = None
                return self.hidden(x, attn, past, cache, train)
            raise
        return out.last_hidden_state, (out.past_key_values if cache else None)

    def loss(self, batch):
        """Training loss of a batch (operator layers, _op_loss). The values in `parts` are tensors on the device
        (no synchronisation); call float() on them when they are logged."""
        return self._op_loss(batch)

    def _op_loss(self, batch):
        """Operator layers: likelihood of the answer under the mixture of the language model and the registers,
        plus the loss of the pointers of the operator layers (distant supervision)."""
        labels, ob = batch["labels"], batch["op"]
        dev = labels.device
        c = self.op.begin(ob, self.alg, dev, teacher=True)
        self._ctx = c
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=labels.is_cuda):
                h, _ = self.hidden(self.embed(batch["ids"], c), batch["attn"], train=True)
                sel = labels[:, 1:] != -100
                tgt = labels[:, 1:][sel]
                hs = h[:, :-1][sel]
                logits = self.lm_head(hs)
            Kc = self.op.copy_keys(h, c) if (self.op.copy_on or self.op.A) else None
            dptr = dhit = None
            if self.op.A and "dyn" in ob:                       # operations on the numbers of the answer
                Kc = self.op.dyn_train(h, c, ob["dyn"], Kc)
                dptr, dhit = self.op.dyn_loss(c)
            mix, stats = self.op.mix_loss(hs, logits, tgt, sel.nonzero(as_tuple=True)[0], ob["nt"].to(dev)[:, 1:][sel],
                                          ob["used"].to(dev)[:, 1:][sel], Kc)
            ptr, hit = self.op.pointer_loss(c)
        finally:
            self._ctx = None
        lm_loss = stats.pop("lm_loss")                          # the language model alone, with gradient
        parts = {"mix": mix.detach(), **stats}               # stats["lm"]: the language model alone (no second copy
        total = mix                                             # of the logits: memory)
        if self.w_lm > 0:                                       # the model keeps writing every token itself (fallback
            total = total + self.w_lm * lm_loss                 # when no register holds the number)
        if ptr is not None:
            total = total + self.w_ptr * ptr
            parts.update(ptr=ptr.detach(), ptr_acc=hit.detach())
        if dptr is not None:
            total = total + self.w_ptr * dptr
            parts.update(dptr=dptr.detach(), dptr_acc=dhit.detach())
        return total, parts

    @torch.no_grad()
    def _generate_op(self, examples, max_new):
        """Greedy decoding with operator layers. The operator layers run once, on the prompt; then every token is
        the arg max of the mixture of the language model and the registers. op_mode 'off': the same model without
        operator layers and without the mixture; 'shuffle': the vectors of other numbers (causal tests)."""
        from oplayers import next_char, num_prefix, number_ended, written
        dev = next(self.model.parameters()).device
        rows = self.encode_examples(examples, with_answer=False)
        T, B, S = max(len(r[0]) for r in rows), len(rows), self.op.S_alloc
        ids = torch.tensor([[self.pad_id] * (T - len(r[0])) + r[0] for r in rows], device=dev)
        attn = torch.tensor([[0] * (T - len(r[0])) + [1] * len(r[0]) for r in rows], device=dev)
        c = None
        if self.op_mode != "off":
            c = self.op.begin(self._op_batch(rows, [T - len(r[0]) for r in rows], T, teacher=False), self.alg, dev,
                              teacher=False, shuffle=self.op_mode == "shuffle")
        self._ctx = c
        E = self.model.get_input_embeddings()
        done, out_ids, texts = [False] * B, [[] for _ in range(B)], [""] * B
        strings, Kc = None, None
        self.op_regs = None                                              # the register strings of every example
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=ids.is_cuda):
                full = self.embed(ids, c)
                cur, past = full, None
                for step in range(max_new):
                    if c is not None and step > 0:
                        c.phase = "skip" if self.use_cache else "replay"
                    h, past = self.hidden(cur if self.use_cache else full, attn, past, cache=self.use_cache)
                    hl = h[:, -1]
                    logits = self.lm_head(hl)
                    if c is not None and step == 0:
                        strings = self._register_strings(c)
                        M = self.op.lay.M
                        self.op_regs = [{"numbers": [v for s, v in sorted(row.items()) if s < M],
                                         "results": [v for s, v in sorted(row.items()) if s >= M]} for row in strings]
                        Kc = None
                        if self.op.copy_on or self.op.A:
                            Kc = self.op.copy_keys(h, c)
                            self.op.dyn_begin(c, Kc)                 # registers of the numbers of the answer
                            Kc = c.K if self.op.copy_on else None
                    copied = None
                    if Kc is not None:
                        nt = torch.full((B, S), -1, dtype=torch.long)
                        used = torch.zeros(B, S, dtype=torch.bool)
                        for b in range(B):
                            if not done[b]:
                                pre, seen = num_prefix(texts[b]), written(texts[b])
                                for s, string in strings[b].items():
                                    ch = next_char(string, pre)
                                    if ch in self.c2id:
                                        nt[b, s] = self.c2id[ch]
                                    if string in seen:
                                        used[b, s] = True
                        nxt, copied = self.op.mix_next(hl, logits, Kc, nt.to(dev), used.to(dev))
                    else:
                        nxt = logits.float().argmax(-1)
                    nl = nxt.tolist()                                        # the one synchronisation of the step
                    cp = copied.tolist() if copied is not None else [False] * B
                    ended_rows, ended_vals = [], []
                    for b in range(B):
                        if done[b]:
                            continue
                        if nl[b] in self.stop_ids:
                            done[b] = True
                        else:
                            prev = texts[b]
                            out_ids[b].append(nl[b])
                            texts[b] += self._tok_string(nl[b])
                            self.op_log["tokens"] += 1
                            self.op_log["copied"] += int(cp[b])
                            if c is not None and self.op.A:          # a number of the answer has just ended
                                v = number_ended(prev, texts[b])
                                if v is not None:
                                    ended_rows.append(b)
                                    ended_vals.append(v)
                    if ended_rows:                                   # hl[b]: final state of the number's last digit
                        for (b, s), st in self.op.dyn_step(c, ended_rows, ended_vals, hl).items():
                            strings[b][s] = st
                            if s >= self.op.S_total or (s >= self.op.lay.S and (s - self.op.lay.S) % 3):
                                self.op_regs[b]["results"].append(st)
                        if self.op.copy_on:
                            Kc = c.K
                    if all(done):
                        break
                    cur = E(torch.tensor([self.pad_id if d else t for t, d in zip(nl, done)], device=dev))[:, None]
                    attn = torch.cat([attn, attn.new_ones(B, 1)], 1)
                    if not self.use_cache:
                        full = torch.cat([full, cur], 1)
        finally:
            self._ctx = None
        return [(self.tok.decode(o, skip_special_tokens=True), []) for o in out_ids]

    @torch.no_grad()
    def sanity_nll(self):
        """Mean negative log-likelihood (nats per token) of an English sentence under the base model, and the
        largest difference between the logits of the complete model and those of decoder + output layer as used
        here. A model whose weights were not loaded gives about log(vocabulary) = 12; a trained one a few nats."""
        text = ("The capital of France is Paris. Water freezes at zero degrees Celsius and boils at one hundred "
                "degrees. There are seven days in a week and twelve months in a year.")
        ids = torch.tensor([self.tok(text, add_special_tokens=False)["input_ids"]], device=next(self.model.parameters()).device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=ids.is_cuda):
            ref = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), return_dict=True).logits[0].float()
            h, _ = self.hidden(self.model.get_input_embeddings()(ids), torch.ones_like(ids))
            own = self.lm_head(h[0]).float()
        diff = (ref - own).abs().max().item() / max(1e-6, ref.abs().max().item())
        return F.cross_entropy(own[:-1], ids[0, 1:]).item(), diff

    # ------------------------------------------------------------------------------- generation
    @torch.no_grad()
    def generate(self, examples, max_new):
        """Greedy decoding (sampling with self.temperature > 0). Shorter prompts are padded on the left and masked
        (checked for this architecture: the outputs do not change). Returns for every example the generated text and
        an empty list. One device synchronisation per generated token."""
        if self.op is not None:
            return self._generate_op(examples, max_new)
        dev = next(self.model.parameters()).device
        rows = self.encode_examples(examples, with_answer=False)
        T = max(len(r[0]) for r in rows)
        ids = torch.tensor([[self.pad_id] * (T - len(r[0])) + r[0] for r in rows], device=dev)
        attn = torch.tensor([[0] * (T - len(r[0])) + [1] * len(r[0]) for r in rows], device=dev)
        B = ids.shape[0]
        E = self.model.get_input_embeddings()
        done = [False] * B
        out_ids = [[] for _ in range(B)]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=ids.is_cuda):
            full = self.embed(ids)
            cur, past = full, None
            for step in range(max_new):
                h, past = self.hidden(cur if self.use_cache else full, attn, past, cache=self.use_cache)
                hl = h[:, -1]
                lg = self.lm_head(hl).float()
                if getattr(self, "temperature", 0.0) > 0:             # sampling (make_solutions.py); greedy otherwise
                    nxt = torch.multinomial(torch.softmax(lg / self.temperature, -1), 1, generator=self.sample_gen).squeeze(1)
                else:
                    nxt = lg.argmax(-1)
                nl = nxt.tolist()                                            # the one synchronisation of the step
                emb = E(torch.tensor([self.pad_id if d else t_ for t_, d in zip(nl, done)], device=dev))
                for b in range(B):
                    if done[b]:
                        continue
                    if nl[b] in self.stop_ids:
                        done[b] = True
                    else:
                        out_ids[b].append(nl[b])
                if all(done):
                    break
                cur = emb[:, None]
                attn = torch.cat([attn, attn.new_ones(B, 1)], 1)
                if not self.use_cache:
                    full = torch.cat([full, cur], 1)
        return [(self.tok.decode(o, skip_special_tokens=True), []) for o in out_ids]

    @torch.no_grad()
    def check_cache(self, examples, max_new=6):
        """Use the key-value cache only if it reproduces decoding without a cache on a few prompts."""
        try:
            self.use_cache = True
            a = self.generate(examples, max_new)
            self.use_cache = False
            b = self.generate(examples, max_new)
            self.use_cache = a == b
            return "cache agrees with full recomputation" if self.use_cache else "cache DISAGREES: decoding without cache"
        except Exception as e:                           # the cache of this architecture cannot be used this way
            self.use_cache = False
            return f"cache unusable ({type(e).__name__}: {str(e)[:120]}): decoding without cache"
