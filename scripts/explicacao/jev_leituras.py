#!/usr/bin/env python3
"""Jev readings of the explanation front after the R1 failure: one-sided bounds on the occlusion
effect, validation 4 (options) and validation 2 (deletion and insertion) on the rounded Jev games
(preregistration/implementation_notes.md, Jev readings after the R1 failure).

Usage: jev_leituras.py limites | val4 | val2 [--workers 6]
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import binomtest, kendalltau  # noqa: E402

import concorrentes as C  # noqa: E402
import games as G  # noqa: E402
from g3_f1 import banzhaf  # noqa: E402
from g5 import deletion_percentile  # noqa: E402
from r1 import design, gibbs, intervals, prior_diag, wi_ei  # noqa: E402
from val2 import METHODS, insertion_percentile  # noqa: E402
from val4 import max_set  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_items  # noqa: E402

DER = Path(__file__).resolve().parents[2] / "data/derived/explicacao"
CASES = (("pt", "jev-1.13.0"), ("pt", "jev-1.14.0"), ("en", "jev-1.13.0"), ("en", "jev-1.14.0"))
KINDS = ("original", "alt1", "alt2", "placebo")


def case_name(loc, ver):
    return f"jev_{loc}_{ver[4:8]}"


def table(probs, d):
    return np.array([probs[f"c{m}"] for m in range(1 << d)], float)


def games():
    """(loc, item) -> (d, options, {version: rounded probabilities})."""
    out = {}
    for line in open(DER / "jev_games.jsonl"):
        r = json.loads(line)
        out[(r["loc"], r["item"])] = (r["d"], r["options"], {v: table(x["probs"], r["d"]) for v, x in r["versions"].items()})
    return out


def top2(ph_full):
    o = np.argsort(-ph_full, kind="stable")
    return int(o[0]), int(o[1])


# --- one-sided bounds -------------------------------------------------------------------------

def bounds(ph, d):
    full = (1 << d) - 1
    j = int(np.argmax(ph[full]))
    lo, hi = intervals(ph[:, j])
    low = np.array([lo[full] - hi[full ^ (1 << i)] for i in range(d)])
    up = np.array([hi[full] - lo[full ^ (1 << i)] for i in range(d)])
    return j, low, up


def limites():
    g = games()
    res = {}
    for loc in ("pt", "en"):
        _, gold = load_items(loc)
        per = {}
        for ver in ("jev-1.13.0", "jev-1.14.0"):
            units, items_pos, items_top, widths, dec, excl = [], [], [], [], {}, 0
            for (l, item), (d, _, P) in g.items():
                if l != loc:
                    continue
                if d < 2:
                    excl += 1
                    continue
                j, low, up = bounds(P[ver], d)
                dec[item] = (j, low, up)
                fl, fu = np.isfinite(low), np.isfinite(up)
                units += [(a, b, c) for a, b, c in zip(fl, fu, low > 0)]
                items_pos.append(bool((low > 0).any()))
                k = int(np.argmax(low))
                items_top.append(bool(low[k] >= np.max(np.delete(up, k))))
                widths += list((up - low)[fl & fu])
            u = np.array(units)
            per[ver] = {"n_itens": len(items_pos), "excluidos_d1": excl, "n_unidades": len(u),
                        "dois_lados": round(float((u[:, 0] & u[:, 1]).mean()), 4),
                        "so_inferior": round(float((u[:, 0] & ~u[:, 1]).mean()), 4),
                        "so_superior": round(float((~u[:, 0] & u[:, 1]).mean()), 4),
                        "nenhum": round(float((~u[:, 0] & ~u[:, 1]).mean()), 4),
                        "unidades_W_certamente_positivo": round(float(u[:, 2].mean()), 4),
                        "itens_com_unidade_certa": round(float(np.mean(items_pos)), 4),
                        "itens_topo_identificado": round(float(np.mean(items_top)), 4),
                        "largura_mediana_finita": round(float(np.median(widths)), 3) if widths else None,
                        "_dec": dec}
        a, b = per["jev-1.13.0"].pop("_dec"), per["jev-1.14.0"].pop("_dec")
        common = sorted(set(a) & set(b))
        ok13 = np.array([a[i][0] == gold[i] for i in common])
        ok14 = np.array([b[i][0] == gold[i] for i in common])
        n10, n01 = int((ok13 & ~ok14).sum()), int((~ok13 & ok14).sum())
        disj = [bool(a[i][1][u] > b[i][2][u] or b[i][1][u] > a[i][2][u]) for i in common for u in range(len(a[i][1]))]
        per["entre_versoes"] = {"n_itens": len(common), "decisao_igual": round(float(np.mean([a[i][0] == b[i][0] for i in common])), 4),
                                "acerto_1.13": round(float(ok13.mean()), 4), "acerto_1.14": round(float(ok14.mean()), 4),
                                "mcnemar": {"so_1.13": n10, "so_1.14": n01, "p": float(binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0},
                                "unidades_conjuntos_disjuntos": round(float(np.mean(disj)), 4)}
        res[loc] = per
    (DER / "jev_limites.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))


# --- validation 4 -----------------------------------------------------------------------------

def jk_intervals(ph, j, k):
    pj, pk = ph[:, j], ph[:, k]
    with np.errstate(divide="ignore"):
        lo = np.where(pj <= 0, -np.inf, np.log(np.maximum(pj - 0.005, 1e-12))) - np.log(np.minimum(pk + 0.005, 1.0))
        hi = np.log(np.minimum(pj + 0.005, 1.0)) - np.where(pk <= 0, -np.inf, np.log(np.maximum(pk - 0.005, 1e-300)))
    return lo, hi


def w_post(args):
    ph, d, j, k, seed = args
    X, pairs = design(d)
    lo, hi = jk_intervals(ph, j, k)
    draws = gibbs(X, lo, hi, prior_diag(d, 2.0, 1.0), np.random.default_rng(np.random.SeedSequence(seed)))
    return np.array([wi_ei(draws, d, pairs, i)[0].mean() for i in range(d)])


def val4(workers):
    g = games()
    rows = [json.loads(line) for line in open(DER / "jev_validations.jsonl")]
    res = {}
    for cid, (loc, ver) in enumerate(CASES):
        sel = {(r["item"], r["kind"]): r for r in rows if r["loc"] == loc and r["version"] == ver}
        items = sorted({i for i, _ in sel})
        tasks, meta = [], []
        for item in items:
            d, opts, P = g[(loc, item)]
            j, k = sel[(item, "alt1")]["j"], sel[(item, "alt1")]["k_star"]
            for c, kind in enumerate(KINDS):
                if kind == "original":
                    ph, opt = P[ver], opts
                else:
                    ph, opt = table(sel[(item, kind)]["probs"], d), sel[(item, kind)]["options"]
                jj, kk = (opt.index(opts[j]), opt.index(opts[k])) if kind == "placebo" else (j, k)
                valid = top2(ph[(1 << d) - 1]) == (jj, kk)
                meta.append((item, kind, valid))
                tasks.append((ph, d, jj, kk, [20261021, 65, cid, item, c]))
        with ProcessPoolExecutor(workers) as ex:
            ws = list(ex.map(w_post, tasks, chunksize=4))
        by = {(m[0], m[1]): (m[2], w) for m, w in zip(meta, ws)}
        rng = np.random.default_rng(np.random.SeedSequence([20261021, 66, cid]))
        out = {"n_itens": len(items), "original_valido": round(float(np.mean([by[(i, "original")][0] for i in items])), 4)}
        for alt in ("alt1", "alt2"):
            ok = [i for i in items if by[(i, alt)][0] and by[(i, "placebo")][0]]
            a0 = {i: max_set(by[(i, "original")][1]) for i in ok}
            ca = np.array([len(a0[i] & max_set(by[(i, alt)][1])) == 0 for i in ok], float)
            cp = np.array([len(a0[i] & max_set(by[(i, "placebo")][1])) == 0 for i in ok], float)
            if len(ok) == 0:
                out[alt] = {"n_validos": 0}
                continue
            D = float(ca.mean() - cp.mean())
            boot = [float(ca[s].mean() - cp[s].mean()) for s in (rng.integers(0, len(ok), len(ok)) for _ in range(10000))]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            cls = ("resultado central" if D >= 0.10 and lo > 0 else "dependência detectável" if D >= 0.02 and lo > 0 else "invariância aproximada")
            out[alt] = {"n_validos": len(ok), "excluidos_frac": round(1 - len(ok) / len(items), 4), "mudanca_alt": round(float(ca.mean()), 4),
                        "mudanca_placebo": round(float(cp.mean()), 4), "D": round(D, 4), "IC95": [round(float(lo), 4), round(float(hi), 4)], "leitura": cls}
        res[case_name(loc, ver)] = out
        print(case_name(loc, ver), json.dumps(out, ensure_ascii=False), flush=True)
    (DER / "val4_jev.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))


# --- validation 2 -----------------------------------------------------------------------------

def val2():
    g = games()
    rows = [json.loads(line) for line in open(DER / "jev_validations.jsonl")]
    res = {}
    for cid, (loc, ver) in enumerate(CASES):
        ops = {r["item"]: r for r in rows if r["loc"] == loc and r["version"] == ver and r["kind"] == "operator"}
        acc = {(m, op, cv): [] for m in METHODS for op in ("remocao", "troca") for cv in ("morf", "insercao")}
        for item, r in sorted(ops.items()):
            d, _, P = g[(loc, item)]
            if d < 2:
                continue
            ph = P[ver]
            j = int(np.argmax(ph[(1 << d) - 1]))
            v = G.value(ph, j)
            vop = G.value(table(r["probs"], d), j)
            F, W = G.first_last(v, d)
            masks = C.kernel_samples(d, np.random.default_rng(np.random.SeedSequence([20261021, 86, 2 + cid, item])))
            att = {"shapley": G.shapley(G.mobius(v), d), "W": W, "F": F, "kernelshap": C.kernelshap_cl(v, d, masks)[0],
                   "bayesshap": C.bayesshap(ph[:, j], d, masks)[0], "hedge": np.array([banzhaf(v, d, i) for i in range(d)])}
            for m, s in att.items():
                order = np.argsort(-np.asarray(s), kind="stable")
                for op, game in (("remocao", v), ("troca", vop)):
                    acc[(m, op, "morf")].append(deletion_percentile(game, d, order))
                    acc[(m, op, "insercao")].append(insertion_percentile(game, d, order))
        np.savez_compressed(DER / f"val2_{case_name(loc, ver)}_itens.npz", **{"|".join(k): np.array(v) for k, v in acc.items()})
        out = {"n_itens": len(acc[("shapley", "remocao", "morf")]), "media_u": {}, "kendall": {}}
        for cv in ("morf", "insercao"):
            means = {op: {m: float(np.mean(acc[(m, op, cv)])) for m in METHODS} for op in ("remocao", "troca")}
            out["media_u"][cv] = {op: {m: round(x, 4) for m, x in mm.items()} for op, mm in means.items()}
            tau = kendalltau([means["remocao"][m] for m in METHODS], [means["troca"][m] for m in METHODS])[0]
            out["kendall"][cv] = {"tau": round(float(tau), 4), "ordena": bool(tau >= 0.6),
                                  "ranking_remocao": sorted(METHODS, key=lambda m: -means["remocao"][m])}
        res[case_name(loc, ver)] = out
        print(case_name(loc, ver), json.dumps(out["kendall"], ensure_ascii=False), flush=True)
    (DER / "val2_jev.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("limites", "val4", "val2"))
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    {"limites": limites, "val4": lambda: val4(a.workers), "val2": val2}[a.what]()


if __name__ == "__main__":
    main()
