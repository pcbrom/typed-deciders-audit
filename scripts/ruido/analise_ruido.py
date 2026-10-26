#!/usr/bin/env python3
"""Analysis of robustness to natural noise (implementation_notes.md): paired accuracy against clean,
decisions that change, ECE after the clean temperature, and the gate calibrated on clean items applied
to noisy report items; plus training with noise against the G4 original heads. Output:
data/derived/ruido/analise_ruido.json.

Usage: analise_ruido.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import betaincinv

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))
import ruido as R  # noqa: E402
from ablation_gate import beta_cfg, threshold  # noqa: E402
from common import DATA, ece, fit_temperature, load_items, load_systems, log_probs, softmax_t  # noqa: E402

D = DATA / "derived" / "ruido"
RULES = {"beta": beta_cfg(0.05, "uniform", "ascending"), "ltt": {"rule": "ltt", "delta": 0.05, "min_n": 20}}


def paired(d, seed):
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
    return {"media": round(float(d.mean()), 4), "IC95": [round(float(x), 4) for x in np.percentile(boot, [2.5, 97.5])]}


def gate(conf_cal, ok_cal, conf_pop, ok_pop, seed):
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    out = {}
    for t in (0.95, 0.99):
        for r, cfg in RULES.items():
            v, n = 0, 1000
            for _ in range(n):
                s = rng.integers(0, len(ok_cal), len(ok_cal))
                tau = threshold(cfg, conf_cal[s], ok_cal[s], t)
                if tau is not None:
                    acc = conf_pop >= tau
                    v += bool(acc.any() and ok_pop[acc].mean() < t)
            out[f"{r}_{t}"] = {"violacao": round(v / n, 4), "sup95": round(1.0 if v == n else float(betaincinv(v + 1, n - v, 0.95)), 4)}
    return out


def systems(loc):
    """Calibration probabilities (clean, items 0 to 999) and report probabilities per condition."""
    _, gold = load_items(loc)
    out = {}
    for m in ("multilingual", "ft"):
        z = np.load(D / f"laya_{m}_{loc}.npz")
        lp = log_probs(z["limpo_todos"])
        T = fit_temperature(lp[:1000], gold[:1000])
        rep = {"limpo": z["limpo_todos"][1000:]}
        rep.update({c: z[c] for c in z.files if c != "limpo_todos"})
        out[f"laya_{m}"] = (softmax_t(lp[:1000], T), {c: softmax_t(log_probs(P), T) for c, P in rep.items()})
    rows = [json.loads(line) for line in open(D / "jev.jsonl")]
    jev = {}
    for r in rows:
        if r["loc"] == loc:
            jev.setdefault(r["condition"], {})[r["item"]] = r["probs"]
    cal = load_systems(loc)["jev"][:1000]
    rep = {c: np.array([v[i] for i in range(1000, 2000)], float) for c, v in jev.items() if len(v) == 1000}
    out["jev"] = (cal, rep)
    return out, gold


def main():
    res = {"robustez": {}, "treino": {}}
    for loc in ("pt", "en"):
        S, gold = systems(loc)
        g, gc = gold[1000:], gold[:1000]
        for k, (cal, rep) in enumerate(S.items()):
            name, (Pc, reps) = cal, rep
            if "limpo" not in reps:
                continue
            ok0 = reps["limpo"].argmax(1) == g
            cc, okc = Pc.max(1), Pc.argmax(1) == gc
            base_gate = gate(cc, okc, reps["limpo"].max(1), ok0, [20261023, 12, k, loc == "en"])
            row = {"limpo": {"acc": round(float(ok0.mean()), 4), "ece": round(float(ece(reps["limpo"].max(1), ok0.astype(float))), 4), "portao": base_gate}}
            for c in R.conditions(loc)[1:]:
                if c not in reps:
                    continue
                P = reps[c]
                ok = P.argmax(1) == g
                pg = gate(cc, okc, P.max(1), ok, [20261023, 12, k, loc == "en"])
                broken = {r: bool(base_gate[r]["sup95"] <= 0.05 and pg[r]["sup95"] > 0.05) for r in pg}
                row[c] = {"acc": round(float(ok.mean()), 4), "dif_acc": paired(ok.astype(float) - ok0, [20261023, 11]),
                          "decisao_muda": round(float((P.argmax(1) != reps["limpo"].argmax(1)).mean()), 4),
                          "ece": round(float(ece(P.max(1), ok.astype(float))), 4), "portao": pg, "promessa_quebrada_pelo_ruido": broken}
            res["robustez"][f"{name}_{loc}"] = row
            print(name, loc, {c: (v["acc"], v.get("dif_acc", {}).get("IC95")) for c, v in row.items()}, flush=True)
    files = sorted(D.glob("treino_ruido_s*.npz"))
    if files:
        for loc in ("pt", "en"):
            _, gold = load_items(loc)
            g = gold[1000:]
            for c in R.conditions(loc):
                d = []
                for f in files:
                    s = f.stem.split("_s")[-1]
                    A = np.load(f)[f"{loc}|{c}"].argmax(1) == g
                    O = np.load(D / f"treino_original_s{s}.npz")[f"{loc}|{c}"].argmax(1) == g
                    d.append(A.astype(float) - O)
                d = np.mean(d, 0)
                p = paired(d, [20261023, 15])
                entry = {"sementes": len(files), "dif_acc": p}
                if c == "limpo":
                    entry["nao_inferior_margem_2"] = bool(p["IC95"][0] > -0.02)
                if c == "misto_0.10":
                    entry["superior"] = bool(p["IC95"][0] > 0)
                res["treino"][f"{loc}|{c}"] = entry
        pt = res["treino"]
        res["treino"]["ganho_protocolo_pt"] = bool(pt.get("pt|limpo", {}).get("nao_inferior_margem_2") and pt.get("pt|misto_0.10", {}).get("superior"))
        res["treino"]["ganho_protocolo_en"] = bool(pt.get("en|limpo", {}).get("nao_inferior_margem_2") and pt.get("en|misto_0.10", {}).get("superior"))
        print({k: v for k, v in res["treino"].items()}, flush=True)
    (D / "analise_ruido.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
