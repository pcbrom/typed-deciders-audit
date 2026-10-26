#!/usr/bin/env python3
"""Gate G4 of the explanation front: additive head on Laya against the original head, same recipe and
seeds (implementation_notes.md, G4 implementation).

z_j = b_j + sum_{t in state} LN(h_t)' U V' LN(m~_j), with m~_j and b_j = scorer(m~_j) from a pass of the
same network with an empty state, so the state enters only through the sum.

Run with the interpreter of arquivos/modelos_decisao/.venv.
Usage: aditiva.py --seed 0 [--head additive|original]
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "drift"))
from common import DATA, load_items, EXTERNAL  # noqa: E402
from data import make_item, massive  # noqa: E402
from laya.common import collate_items  # noqa: E402
from model import Learner  # noqa: E402

WEIGHTS = (EXTERNAL / "laya-ajustado/aditiva")
OUT = DATA / "derived" / "explicacao"
RANK = 64


def hidden(model, b, dev):
    """Encoder plus head layers, as in DecisionModel.forward, returning token states."""
    h = model.encoder(input_ids=b["input_ids"].to(dev), attention_mask=b["attention_mask"].to(dev)).last_hidden_state
    h = h + model.type_emb(b["qtype"].to(dev))[:, None, :]
    if model.head is not None:
        pad = ~b["attention_mask"].to(dev).bool()
        for layer in model.head.layers:
            h = layer(h, src_key_padding_mask=pad)
    return h


def state_mask(ids, att, sep_id):
    """Positions after the separator that closes the options, the final separator included."""
    is_sep = ids == sep_id
    cum = torch.cumsum(is_sep.long(), 1)
    return (cum >= 2) & ~(is_sep & (cum == 2)) & att.bool()


class Additive(nn.Module):
    def __init__(self, base, d, residual=False):
        super().__init__()
        self.residual = residual
        self.base = base
        self.ln_h, self.ln_m = nn.LayerNorm(d), nn.LayerNorm(d)
        self.U, self.V = nn.Linear(d, RANK, bias=False), nn.Linear(d, RANK, bias=False)
        nn.init.normal_(self.U.weight, std=0.02)
        nn.init.normal_(self.V.weight, std=0.02)

    def evidence(self, full, empty, sep_id, dev):
        """Per-token evidence e_{t,j} (batch, tokens, options) and the stateless bias b_j."""
        h = hidden(self.base, full, dev)
        he = hidden(self.base, empty, dev)
        idx = empty["marker_pos"].to(dev).clamp(min=0)[:, :, None].expand(-1, -1, he.size(-1))
        m = torch.gather(he, 1, idx)
        b = self.base.scorer(m).squeeze(-1).float()
        e = torch.einsum("btr,bkr->btk", self.U(self.ln_h(h)).float(), self.V(self.ln_m(m)).float())
        sm = state_mask(full["input_ids"].to(dev), full["attention_mask"].to(dev), sep_id)
        return e * sm[:, :, None], b

    def residual_term(self, full, dev):
        h = hidden(self.base, full, dev)
        idx = full["marker_pos"].to(dev).clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        return self.base.scorer(torch.gather(h, 1, idx)).squeeze(-1).float()

    def forward(self, full, empty, sep_id, dev, with_parts=False):
        e, b = self.evidence(full, empty, sep_id, dev)
        r = self.residual_term(full, dev) if self.residual else torch.zeros_like(b)
        z = (b + e.sum(1) + r).masked_fill(~full["marker_mask"].to(dev), -1e4)
        return (z, e.sum(1), r) if with_parts else z


def train_items():
    rows = [r for r in massive("pt-PT") if r["partition"] == "train"]
    intents = sorted({r["intent"] for r in rows})
    return [make_item(r, intents) for r in rows]


def encode_pair(L, it):
    full = L.encode(it)
    empty = L.encode({**it, "id": it["id"] + ":vazio", "text": ""})
    return full, empty


def finetune_additive(L, A, items, seed, epochs=3, batch=32, lr_encoder=2.5e-5, lr_head=1e-4, l2=0.0):
    dev = L.device
    sep = L.agent.tok.sep_token_id
    pad = L.agent.tok.pad_token_id
    enc = []
    for it in items:
        f, e = encode_pair(L, it)
        f = dict(f); f["target"] = [1.0 if j == it["gold"] else 0.0 for j in range(len(it["options"]))]
        enc.append((f, e))
    groups = [{"params": [p for n, p in A.named_parameters() if n.startswith("base.encoder.") and p.requires_grad], "lr": lr_encoder},
              {"params": [p for n, p in A.named_parameters() if not n.startswith(("base.encoder.", "base.act_head.")) and p.requires_grad],
               "lr": lr_head}]
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    total = max(1, math.ceil(len(enc) / batch) * epochs)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    rng = random.Random(seed)
    A.train()
    t0 = time.perf_counter()
    for _ in range(epochs):
        data = sorted(enc, key=lambda x: len(x[0]["ids"]))
        batches = [data[i:i + batch] for i in range(0, len(data), batch)]
        rng.shuffle(batches)
        for bt in batches:
            full = collate_items([[f] for f, _ in bt], pad)
            empty = collate_items([[e] for _, e in bt], pad)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                z, _, r = A(full, empty, sep, dev, with_parts=True)
            loss = -(full["target"].to(dev) * torch.log_softmax(z.float(), -1)).sum(-1).mean()
            if A.residual:
                mm = full["marker_mask"].to(dev).float()
                loss = loss + l2 * ((r * mm) ** 2).sum() / mm.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
            opt.step()
            sched.step()
    A.eval()
    return time.perf_counter() - t0


@torch.no_grad()
def predict_additive(L, A, items, bs=64):
    dev, sep, pad = L.device, L.agent.tok.sep_token_id, L.agent.tok.pad_token_id
    out = np.zeros((len(items), len(items[0]["options"])), dtype=np.float32)
    pairs = [encode_pair(L, it) for it in items]
    order = np.argsort([len(f["ids"]) for f, _ in pairs], kind="stable")
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        full = collate_items([[pairs[j][0]] for j in idx], pad)
        empty = collate_items([[pairs[j][1]] for j in idx], pad)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            z = A(full, empty, sep, dev)
        out[idx] = torch.softmax(z.float(), -1).cpu().numpy()
    return out


def eval_items(loc):
    items, gold = load_items(loc)
    return [{"id": f"{loc}:{i}", "text": it["utterance"], "options": it["options"], "gold": int(gold[i])} for i, it in enumerate(items)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--head", choices=("additive", "original", "residual"), required=True)
    ap.add_argument("--l2", type=float, default=0.0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    L = Learner("multilingual")
    items = []
    for it in train_items():
        try:
            L.encode(it)
            if a.head != "original":
                encode_pair(L, it)
            items.append(it)
        except ValueError:
            pass
    dropped = 11514 - len(items)
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    if a.head == "original":
        secs = L.finetune(items, epochs=3, batch=32, seed=a.seed)
        torch.save(L.state(), WEIGHTS / f"original_s{a.seed}.pt")
        predict = lambda its: L.predict(its)
    else:
        A = Additive(L.model, L.model.encoder.config.hidden_size, residual=a.head == "residual").to(L.device)
        for p in L.model.act_head.parameters():
            p.requires_grad_(False)
        secs = finetune_additive(L, A, items, a.seed, l2=a.l2)
        tag = f"additive_s{a.seed}" if a.head == "additive" else f"residual_l2{a.l2}_s{a.seed}"
        torch.save(A.state_dict(), WEIGHTS / f"{tag}.pt")
        predict = lambda its: predict_additive(L, A, its)
    res = {"seed": a.seed, "head": a.head, "l2": a.l2, "train_s": round(secs, 1), "descartados_orcamento": dropped}
    arrays = {}
    for loc in ("pt", "en"):
        its = eval_items(loc)
        P = predict(its)
        arrays[f"{loc}_probs"] = P
        g = np.array([it["gold"] for it in its])
        res[f"acc_relato_{loc}"] = round(float((P[1000:].argmax(1) == g[1000:]).mean()), 4)
    if a.head == "residual":
        dev, sep, pad = L.device, L.agent.tok.sep_token_id, L.agent.tok.pad_token_id
        its = eval_items("pt")[1000:]
        shares = []
        with torch.no_grad():
            for i in range(0, len(its), 64):
                pairs = [encode_pair(L, x) for x in its[i:i + 64]]
                full = collate_items([[f] for f, _ in pairs], pad); empty = collate_items([[e] for _, e in pairs], pad)
                _, es, r = A(full, empty, sep, dev, with_parts=True)
                ve, vr = es.float().var(1), r.float().var(1)
                shares += (ve / (ve + vr + 1e-12)).cpu().tolist()
        res["parcela_aditiva_mediana"] = round(float(np.median(shares)), 4)
    name = f"g4_{a.head}_s{a.seed}" if a.head != "residual" else f"g4_residual_l2{a.l2}_s{a.seed}"
    np.savez_compressed(OUT / f"{name}.npz", **arrays)
    (OUT / f"{name}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
