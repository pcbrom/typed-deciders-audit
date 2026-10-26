#!/usr/bin/env python3
"""Gate G4 of the explanation front: accuracy, calibration and the author's Beta gate of the additive
head against the original head, three seeds, pt report items and en as transfer
(preregistration/explicacao_portoes.md, G4, and implementation_notes.md).

Usage: g4_metricas.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
from ablation_gate import beta_cfg, threshold  # noqa: E402
from common import ece, fit_temperature, load_items, log_probs, softmax_t  # noqa: E402

DER = HERE.parents[2] / "data/derived/explicacao"
SEEDS = (0, 1, 2)


def calibrate(P, gold):
    t = fit_temperature(log_probs(P[:1000]), gold[:1000])
    return softmax_t(log_probs(P), t)


def metrics(Pc, gold):
    rep, g = Pc[1000:], gold[1000:]
    ok = rep.argmax(1) == g
    onehot = np.eye(rep.shape[1])[g]
    return {"acc": float(ok.mean()), "nll": float(-np.log(np.clip(rep[np.arange(len(g)), g], 1e-12, 1)).mean()),
            "brier": float(((rep - onehot) ** 2).sum(1).mean()), "ece": float(ece(rep.max(1), ok.astype(float)))}, ok


def gate(Pc, gold, rng, n_part=1000):
    cfg = beta_cfg(0.05, "uniform", "ascending")
    conf, ok = Pc.max(1), Pc.argmax(1) == gold
    dec, broken = [], []
    for _ in range(n_part):
        perm = rng.permutation(len(gold))
        cal, rep = perm[:1000], perm[1000:]
        tau = threshold(cfg, conf[cal], ok[cal], 0.95)
        acc = conf[rep] >= tau if tau is not None else np.zeros(len(rep), bool)
        dec.append(1000 * acc.mean())
        broken.append(bool(acc.any() and ok[rep][acc].mean() < 0.95))
    return {"decisoes_por_1000": round(float(np.mean(dec)), 1), "promessa_quebrada": round(float(np.mean(broken)), 4)}


def main():
    res = {}
    for loc in ("pt", "en"):
        items, gold = load_items(loc)
        rng = np.random.default_rng(np.random.SeedSequence([20261021, 87, loc == "en"]))
        per_seed, oks, eces = {}, {"original": [], "additive": []}, {"original": [], "additive": []}
        for s in SEEDS:
            for h in ("original", "additive"):
                f = DER / f"g4_{h}_s{s}.npz"
                if not f.exists():
                    continue
                Pc = calibrate(np.load(f)[f"{loc}_probs"].astype(float), gold)
                m, ok = metrics(Pc, gold)
                m.update(gate(Pc, gold, rng))
                per_seed[f"{h}_s{s}"] = {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}
                oks[h].append(ok)
                eces[h].append(Pc)
        n = min(len(oks["original"]), len(oks["additive"]))
        if n == 0:
            continue
        O, A = np.array(oks["original"][:n], float), np.array(oks["additive"][:n], float)
        diff_items = (A - O).mean(0)
        brng = np.random.default_rng(np.random.SeedSequence([20261021, 88, loc == "en"]))
        g = gold[1000:]
        boot_acc, boot_ece = [], []
        for _ in range(2000):
            idx = brng.integers(0, len(g), len(g))
            boot_acc.append(diff_items[idx].mean())
            e_a = np.mean([ece(P[1000:][idx].max(1), (P[1000:][idx].argmax(1) == g[idx]).astype(float)) for P in eces["additive"][:n]])
            e_o = np.mean([ece(P[1000:][idx].max(1), (P[1000:][idx].argmax(1) == g[idx]).astype(float)) for P in eces["original"][:n]])
            boot_ece.append(e_a - e_o)
        b = int(((A[0] == 1) & (O[0] == 0)).sum()); c = int(((A[0] == 0) & (O[0] == 1)).sum())
        lo_acc = float(np.percentile(boot_acc, 2.5)) * 100
        ece_ci = [float(x) for x in np.percentile(boot_ece, [2.5, 97.5])]
        verdict = ("competitiva" if lo_acc > -2 and ece_ci[0] >= -0.01 and ece_ci[1] <= 0.01 else
                   "custo medido" if lo_acc > -5 else "resultado negativo")
        res[loc] = {"sementes": n, "por_semente": per_seed,
                    "diferenca_acc_pp": {"media": round(float(diff_items.mean() * 100), 2),
                                         "IC95": [round(float(np.percentile(boot_acc, 2.5)) * 100, 2), round(float(np.percentile(boot_acc, 97.5)) * 100, 2)]},
                    "mcnemar_semente0": {"so_aditiva": b, "so_original": c, "p": float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0},
                    "diferenca_ece_IC95": [round(x, 4) for x in ece_ci], "veredito_G4": verdict}
    (DER / "g4_metricas.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({loc: {k: v for k, v in r.items() if k != "por_semente"} for loc, r in res.items()}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
