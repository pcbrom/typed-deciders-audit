#!/usr/bin/env python3
"""Gate G5 of the explanation front: does the deletion quality of the Shapley order depend on the
model's confidence? Hierarchical model by intent, NUTS (preregistration/explicacao_portoes.md and
implementation_notes.md, G5). Laya cases, exact games.

Usage: g5.py --case pt_multilingual
"""

import argparse
import sys
import itertools
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

import games as G

DER = Path(__file__).resolve().parents[2] / "data/derived/explicacao"
SCRIPTS = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=None)
def orders(d):
    perms = np.array(list(itertools.permutations(range(d))), dtype=np.int64)
    prefix = np.cumsum(1 << perms, axis=1)          # removed set after k steps, as a bit mask
    return perms, prefix


def deletion_percentile(v, d, order):
    full = (1 << d) - 1
    _, prefix = orders(d)
    areas = (v[full] - v[full & ~prefix]).sum(1)
    own_prefix = np.cumsum([1 << int(u) for u in order])
    own = (v[full] - v[full & ~own_prefix]).sum()
    return float(((areas < own - 1e-12).sum() + 0.5 * (np.abs(areas - own) <= 1e-12).sum()) / len(areas))


def laya_rows(case):
    sys.path.insert(0, str(SCRIPTS))
    from common import load_items  # noqa: E402
    z = np.load(DER / f"laya_games_{case}.npz")
    items, _ = load_items(str(z["locale"]))
    rows, skipped = [], 0
    for k in range(len(z["d"])):
        d, item = int(z["d"][k]), int(z["item"][k])
        if d < 2:
            skipped += 1
            continue
        probs = z["probs"][z["offsets"][k]:z["offsets"][k + 1]]
        j = int(np.argmax(probs[(1 << d) - 1]))
        v = G.value(probs, j)
        E = G.shapley(G.mobius(v), d)
        order = np.argsort(-E, kind="stable")
        rows.append({"item": item, "d": d, "u": deletion_percentile(v, d, order), "logit_p0": float(v[(1 << d) - 1]),
                     "intent": items[item]["options"][j]})
    return rows, skipped


JEV_CASES = {"jev_pt_1.13": ("pt", "jev-1.13.0", 0), "jev_pt_1.14": ("pt", "jev-1.14.0", 1),
             "jev_en_1.13": ("en", "jev-1.13.0", 2), "jev_en_1.14": ("en", "jev-1.14.0", 3)}


def _jev_item(args):
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from r1 import design, gibbs, intervals, prior_diag, wi_ei
    item, d, ph, cid = args
    j = int(np.argmax(ph[(1 << d) - 1]))
    X, pairs = design(d)
    lo, hi = intervals(ph[:, j])
    draws = gibbs(X, lo, hi, prior_diag(d, 2.0, 1.0), np.random.default_rng(np.random.SeedSequence([20261021, 64, cid, item])))
    v = G.value(ph, j)
    us = []
    for t in np.linspace(0, len(draws) - 1, 100).astype(int):
        E = np.array([wi_ei(draws[t:t + 1], d, pairs, i)[1][0] for i in range(d)])
        us.append(deletion_percentile(v, d, np.argsort(-E, kind="stable")))
    return {"item": item, "d": d, "u": float(np.mean(us)), "s": float(np.std(us)), "logit_p0": float(v[(1 << d) - 1]), "j": j}


def jev_rows(case):
    from concurrent.futures import ProcessPoolExecutor
    loc, version, cid = JEV_CASES[case]
    tasks, meta, cens = [], {}, 0
    for line in open(DER / "jev_games.jsonl"):
        r = json.loads(line)
        if r["loc"] != loc or r["d"] < 2:
            continue
        d = r["d"]
        ph = np.array([r["versions"][version]["probs"][f"c{m}"] for m in range(1 << d)], float)
        if ph[(1 << d) - 1].max() >= 0.995:
            cens += 1
            continue
        meta[r["item"]] = r["options"]
        tasks.append((r["item"], d, ph, cid))
    with ProcessPoolExecutor(8) as ex:
        rows = list(ex.map(_jev_item, tasks, chunksize=2))
    for row in rows:
        row["intent"] = meta[row["item"]][row.pop("j")]
    return rows, cens


def fit(rows, seed=20261021, warmup=1000):
    import numpyro
    numpyro.set_host_device_count(4)  # before JAX starts, so the four chains run in parallel
    import jax.numpy as jnp
    import numpyro.distributions as dist
    from jax import random
    from numpyro.infer import MCMC, NUTS
    intents = sorted({r["intent"] for r in rows})
    c = jnp.array([intents.index(r["intent"]) for r in rows])
    x = jnp.array([r["logit_p0"] for r in rows]); dd = jnp.array([r["d"] for r in rows], float)
    u = jnp.array([r["u"] for r in rows]); s = jnp.array([r.get("s", 0.0) for r in rows])
    cens = jnp.array([r.get("cens", 0.0) for r in rows]); use_cens = bool(np.any(np.asarray(cens)))

    def model():
        a = numpyro.sample("alpha", dist.Normal(0, 1)); b = numpyro.sample("b", dist.Normal(0, 1))
        eta = numpyro.sample("eta", dist.Normal(0, 1))
        sa = numpyro.sample("sigma_alpha", dist.HalfNormal(0.2)); sb = numpyro.sample("sigma_b", dist.HalfNormal(0.2))
        sig = numpyro.sample("sigma", dist.HalfNormal(1.0))
        with numpyro.plate("intent", len(intents)):  # non-centred: the same model, without the funnel
            ac = a + sa * numpyro.sample("z_alpha", dist.Normal(0, 1)); bc = b + sb * numpyro.sample("z_b", dist.Normal(0, 1))
        mu = ac[c] + bc[c] * x + eta * dd
        if use_cens:
            mu = mu + numpyro.sample("delta", dist.Normal(0, 1)) * cens
        numpyro.sample("u", dist.Normal(mu, jnp.sqrt(s ** 2 + sig ** 2)), obs=u)

    mcmc = MCMC(NUTS(model), num_warmup=warmup, num_samples=1000, num_chains=4, progress_bar=False)
    mcmc.run(random.PRNGKey(seed))
    from numpyro.diagnostics import summary
    summ = summary(mcmc.get_samples(group_by_chain=True))
    rhat = max(float(np.nanmax(np.asarray(v["r_hat"]))) for v in summ.values())
    smp = mcmc.get_samples()
    return smp, rhat, intents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    a = ap.parse_args()
    rows, skipped = jev_rows(a.case) if a.case in JEV_CASES else laya_rows(a.case)
    smp, rhat, intents = fit(rows)
    if rhat > 1.01:
        smp, rhat, intents = fit(rows, warmup=4000)
    b = np.asarray(smp["b"])
    pb = float((b > 0).mean())
    out = {"case": a.case, "n_itens": len(rows), "excluidos": skipped, "excluidos_motivo": "censurados (p0 >= 0,995)" if a.case in JEV_CASES else "d = 1", "r_hat_max": round(rhat, 4),
           "P_b_pos": round(pb, 4), "b_media": round(float(b.mean()), 4), "b_IC90": [round(float(q), 4) for q in np.percentile(b, [5, 95])],
           "sinal_decidido": pb >= 0.95 or pb <= 0.05, "u_medio": round(float(np.mean([r["u"] for r in rows])), 4),
           "spearman_p0_u": round(float(__import__("scipy.stats", fromlist=["spearmanr"]).spearmanr([r["logit_p0"] for r in rows], [r["u"] for r in rows])[0]), 4)}
    (DER / f"g5_{a.case}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    with open(DER / f"g5_{a.case}_itens.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
