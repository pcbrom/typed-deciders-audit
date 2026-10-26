#!/usr/bin/env python3
"""Gate G4, fidelity: on pt report items with d >= 5 units, the Spearman correlation per item between
the additive word scores e_{w,j} - e_{w,k*} and the exact Shapley value of y_jk* in the same additive
model, against a within-item permutation null; references on the original head (occlusion W and SLALOM
against its own exact Shapley); and the share of |e_{t,j} - e_{t,k*}| outside word tokens
(preregistration/explicacao_portoes.md, G4, and implementation_notes.md).

Run with the interpreter of arquivos/modelos_decisao/.venv.
Usage: g4_fidelidade.py --seed 0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "drift"))
from common import DATA, load_items  # noqa: E402
from jev_enumerate import coalition_text, units  # noqa: E402
from laya.common import collate_items, serialize_state  # noqa: E402
from model import Learner  # noqa: E402

import aditiva as AD  # noqa: E402
import games as G  # noqa: E402

OUT = DATA / "derived" / "explicacao"


def unit_of_tokens(L, text, us):
    """Unit index of each state token (or -1 outside the words), from the tokenizer offsets."""
    s = serialize_state({"utterance": text})
    start = s.index(text)
    word_spans, pos = [], 0
    for w in text.split():
        a = text.index(w, pos)
        word_spans.append((start + a, start + a + len(w)))
        pos = a + len(w)
    spans, k = [], 0
    for u in us:
        spans.append((word_spans[k][0], word_spans[k + len(u) - 1][1]))
        k += len(u)
    offs = L.agent.tok(s, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    out = []
    for x, y in offs:
        own = [q for q, (a, b) in enumerate(spans) if x < b and y > a and s[x:y].strip()]
        out.append(own[0] if own else -1)
    return out


def ranks(x):
    from scipy.stats import rankdata
    return rankdata(x)


def mean_rank_corr(pairs):
    """Mean Spearman over items, as Pearson correlation of ranks, skipping constant vectors."""
    vals = []
    for a, b in pairs:
        if np.std(a) > 0 and np.std(b) > 0:
            vals.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(vals))


def permutation_null(rows, n_perm, rng):
    """Mean Spearman when the word scores are permuted within each item, vectorized by d."""
    by_d = {}
    for r in rows:
        a, b = ranks(r["ew"]), ranks(r["shap"])
        if np.std(a) > 0 and np.std(b) > 0:
            by_d.setdefault(len(a), ([], []))
            by_d[len(a)][0].append(a); by_d[len(a)][1].append(b)
    n_items = sum(len(v[0]) for v in by_d.values())
    sums = np.zeros(n_perm)
    for d, (A, B) in by_d.items():
        A, B = np.array(A), np.array(B)
        Bc = (B - B.mean(1, keepdims=True)) / B.std(1, keepdims=True)
        for start in range(0, n_perm, 500):
            m = min(500, n_perm - start)
            perm = np.argsort(rng.random((m, len(A), d)), axis=2)
            Ap = np.take_along_axis(np.broadcast_to(A, (m,) + A.shape), perm, axis=2)
            Ac = (Ap - Ap.mean(2, keepdims=True)) / Ap.std(2, keepdims=True)
            sums[start:start + m] += (Ac * Bc[None]).mean(2).sum(1)
    return sums / n_items


def slalom(v, d, steps=400):
    """SLALOM (Leemann et al., 2024): y(S) ~ c + sum_{t in S} softmax_S(s)_t v_t, least squares on all coalitions."""
    masks = torch.tensor([[m >> i & 1 for i in range(d)] for m in range(1 << d)], dtype=torch.float32)
    y = torch.tensor(v, dtype=torch.float32)
    vv, ss, c = torch.zeros(d, requires_grad=True), torch.zeros(d, requires_grad=True), torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([vv, ss, c], lr=0.05)
    for _ in range(steps):
        w = torch.exp(ss)[None, :] * masks
        f = c + (w * vv[None, :]).sum(1) / w.sum(1).clamp_min(1e-9)
        loss = ((f - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return (torch.softmax(ss, 0) * vv).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--head", choices=("additive", "residual"), default="additive")
    ap.add_argument("--l2", type=float, default=1.0)
    a = ap.parse_args()
    items, gold = load_items("pt")
    sel = [i for i in range(1000, 2000) if len(units(items[i]["utterance"])) >= 5]
    L = Learner("multilingual")
    tag = f"additive_s{a.seed}" if a.head == "additive" else f"residual_l2{a.l2}_s{a.seed}"
    A = AD.Additive(L.model, L.model.encoder.config.hidden_size, residual=a.head == "residual").to(L.device)
    A.load_state_dict(torch.load(AD.WEIGHTS / f"{tag}.pt", map_location=L.device))
    A.eval()
    sep, pad = L.agent.tok.sep_token_id, L.agent.tok.pad_token_id
    rows = []
    for i in sel:
        it, us = items[i], units(items[i]["utterance"])
        d = len(us)
        coal = [{"id": f"f:{i}:{m}", "text": coalition_text(us, m), "options": it["options"], "gold": int(gold[i])} for m in range(1 << d)]
        P = AD.predict_additive(L, A, coal)
        j, k = (int(x) for x in np.argsort(-P[-1], kind="stable")[:2])
        v = G.value(P, j, k)
        shap = G.shapley(G.mobius(v), d)
        full, empty = AD.encode_pair(L, coal[-1])
        with torch.no_grad():
            e, _ = A.evidence(collate_items([[full]], pad), collate_items([[empty]], pad), sep, L.device)
        ids = full["ids"]
        s2 = [p for p, x in enumerate(ids) if x == sep][1]
        diff = (e[0, :, j] - e[0, :, k]).float().cpu().numpy()[:len(ids)]
        owner = unit_of_tokens(L, it["utterance"], us)
        ew = np.zeros(d)
        inside = 0.0
        for t, u in enumerate(owner):
            pos = s2 + 1 + t
            if pos < len(ids) - 1 and u >= 0:
                ew[u] += diff[pos]
                inside += abs(diff[pos])
        total = float(np.abs(diff[s2 + 1:len(ids)]).sum())
        rows.append({"item": i, "d": d, "rho": float(spearmanr(ew, shap)[0]), "fora": 1 - inside / total if total > 0 else 0.0,
                     "ew": ew.tolist(), "shap": shap.tolist()})
        L._enc.clear()
    # original head of the same seed: its own exact game, occlusion and SLALOM
    Lo = Learner("multilingual")
    Lo.load_state(torch.load(AD.WEIGHTS / f"original_s{a.seed}.pt", map_location="cpu"))
    ref = {"oclusao": [], "slalom": []}
    for i in sel:
        it, us = items[i], units(items[i]["utterance"])
        d = len(us)
        coal = [{"id": f"o:{i}:{m}", "text": coalition_text(us, m), "options": it["options"], "gold": int(gold[i])} for m in range(1 << d)]
        P = Lo.predict(coal)
        j, k = (int(x) for x in np.argsort(-P[-1], kind="stable")[:2])
        v = G.value(P, j, k)
        shap = G.shapley(G.mobius(v), d)
        _, W = G.first_last(v, d)
        ref["oclusao"].append(float(spearmanr(W, shap)[0]))
        ref["slalom"].append(float(spearmanr(slalom(v, d), shap)[0]))
        Lo._enc.clear()
    rho = np.array([r["rho"] for r in rows])
    rng = np.random.default_rng(np.random.SeedSequence([20261021, 89]))
    null = permutation_null(rows, 10000, rng)
    p99 = float(np.nanpercentile(null, 99))
    best_ref = max(float(np.nanmean(ref["oclusao"])), float(np.nanmean(ref["slalom"])))
    out = {"seed": a.seed, "head": a.head, "n_itens": len(rows), "spearman_medio_aditiva": round(float(np.nanmean(rho)), 4),
           "nulo_p99": round(p99, 4), "ref_oclusao": round(float(np.nanmean(ref["oclusao"])), 4),
           "ref_slalom": round(float(np.nanmean(ref["slalom"])), 4),
           "fracao_fora_mediana": round(float(np.median([r["fora"] for r in rows])), 4)}
    out["autoexplicacao_por_palavra"] = bool(out["spearman_medio_aditiva"] > p99 and out["spearman_medio_aditiva"] >= best_ref - 0.10
                                             and out["fracao_fora_mediana"] <= 0.2)
    (OUT / (f"g4_fidelidade_s{a.seed}.json" if a.head == "additive" else f"g4_fidelidade_{tag}.json")).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
