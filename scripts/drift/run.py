#!/usr/bin/env python3
"""One run of the drift experiment: an arm on a stream, window by window.

Per window of W items: predict with the current model, gate, collect labels (every escalated
item and a 5% audit of accepted items), add costs, measure the true selective accuracy on the
test split reweighted to the stream mixture, let the arm observe, then retrain, promote and
re-qualify if the arm asks. Of the most recent RETRAIN_N labelled items, the newest HOLDOUT_SHARE
(at least HOLDOUT_MIN) are held out to decide the promotion and the rest, plus 20% replay of the
initial training set, are used to retrain; a retrain needs at least MIN_LABELS labelled items.
Re-qualification uses the REQUAL_N most recent labelled items. These choices are not fixed by the
preregistration and are listed in preregistration/implementation_notes.md.

Usage: run.py --arm author --cfg '{"lam":0.99}' --drift D1 --pattern abrupt --replicate 0 [--role 0]
              [--length 9000] [--out file.jsonl]
"""

import argparse
import json
import time

import numpy as np

import arms as A
import data as D
from model import initial_learner

W, AUDIT, C_E, RETRAIN_N, REQUAL_N, REPLAY = 500, 0.05, 10.0, 4000, 1000, 0.2
HOLDOUT_SHARE, HOLDOUT_MIN, MIN_LABELS = 0.2, 50, 100


def gold(items):
    return np.array([it["gold"] for it in items])


def truth(arm, P_pre, P_post, ok_pre, ok_post, m):
    """Selective accuracy and coverage on the test split reweighted to post-change share m."""
    acc_pre, acc_post = arm.accept(P_pre.max(1)), arm.accept(P_post.max(1))
    w_pre, w_post = (1 - m) / len(acc_pre), m / len(acc_post)
    cov = w_pre * acc_pre.sum() + w_post * acc_post.sum()
    right = w_pre * (acc_pre & ok_pre).sum() + w_post * (acc_post & ok_post).sum()
    return (float(right / cov) if cov > 0 else None), float(cov)


def run(arm_name, cfg, drift, pattern, replicate, role=0, length=D.STREAM_LEN):
    st = D.stream(drift, pattern, replicate, role)
    items = st.items[:length]
    init = D.initial_training(drift)
    L = initial_learner(drift, init)
    arm = A.build(arm_name, cfg)
    rng = np.random.default_rng(np.random.SeedSequence([D.SEED, D.DRIFTS.index(drift), replicate, role, 2]))
    pre_pool = D.pools(drift)[0]
    calib = [pre_pool[i] for i in rng.integers(0, len(pre_pool), 1000)]
    Pc = L.predict(calib)
    arm.initial(Pc.max(1), Pc.argmax(1) == gold(calib))
    g_pre, g_post = gold(st.test_pre), gold(st.test_post)
    P_pre, P_post = L.predict(st.test_pre), L.predict(st.test_post)
    labelled, weights, windows = [], [], []
    totals = {"cost": 0.0, "wrong_accepted": 0, "escalated": 0, "audited": 0, "retrains": 0, "promotions": 0, "gpu_s": 0.0}
    t_start = time.perf_counter()
    for w0 in range(0, len(items), W):
        win = items[w0:w0 + W]
        P = L.predict(win)
        conf, ok = P.max(1), P.argmax(1) == gold(win)
        if hasattr(arm, "prepare"):          # dynamic arms see the window inputs, never its labels, before gating
            arm.prepare(D.features(win))
        acc = arm.accept(conf)
        aud = acc & (rng.random(len(win)) < AUDIT)
        lab = ~acc | aud
        wrong = int((acc & ~ok).sum())
        totals["cost"] += C_E * wrong + int((~acc).sum()) + int(aud.sum())
        totals["wrong_accepted"] += wrong
        totals["escalated"] += int((~acc).sum())
        totals["audited"] += int(aud.sum())
        labelled.extend(it for it, l in zip(win, lab) if l)
        weights.extend(1.0 / AUDIT if a else 1.0 for a, l in zip(aud, lab) if l)   # audited accepted vs escalated
        m = float(st.post_share[w0:w0 + len(win)].mean())
        theta, cov = truth(arm, P_pre, P_post, P_pre.argmax(1) == g_pre, P_post.argmax(1) == g_post, m)
        arm.observe(conf, ok, acc, aud, lab)
        t_now = w0 + len(win)
        rec = {"t": t_now, "post_share": round(m, 3), "theta": theta, "coverage": cov, "stream_coverage": float(acc.mean()),
               "wrong": wrong, "retrain": False, "promoted": False}
        if arm.wants_retrain(t_now) and len(labelled) >= MIN_LABELS:
            recent = labelled[-RETRAIN_N:]
            n_hold = max(HOLDOUT_MIN, int(HOLDOUT_SHARE * len(recent)))
            hold, train = recent[-n_hold:], recent[:-n_hold]
            replay = [init[i] for i in rng.integers(0, len(init), int(REPLAY / (1 - REPLAY) * len(train)))]
            old_state, old_ok = L.state(), L.predict(hold).argmax(1) == gold(hold)
            totals["gpu_s"] += L.finetune(train + replay, epochs=2, seed=int(rng.integers(1 << 30)))
            new_ok = L.predict(hold).argmax(1) == gold(hold)
            promoted = A.promote(arm.promotion, new_ok, old_ok, arm.q)
            if promoted:
                if getattr(arm, "requal_data", "recent") == "holdout":
                    recent, w = hold, np.array(weights[-n_hold:])
                else:
                    recent, w = labelled[-REQUAL_N:], np.array(weights[-REQUAL_N:])
                Pr = L.predict(recent)
                arm.requalify(Pr.max(1), Pr.argmax(1) == gold(recent), w)
                P_pre, P_post = L.predict(st.test_pre), L.predict(st.test_post)
                totals["promotions"] += 1
            else:
                L.load_state(old_state)
            arm.after_retrain(promoted)
            totals["retrains"] += 1
            rec.update(retrain=True, promoted=bool(promoted))
        elif arm.per_window_requalify() and labelled:
            recent = labelled[-REQUAL_N:]
            Pr = L.predict(recent)
            arm.requalify(Pr.max(1), Pr.argmax(1) == gold(recent))
        windows.append(rec)
    post = [r for r in windows if r["post_share"] > 0]
    return {"arm": arm_name, "cfg": cfg, "drift": drift, "pattern": pattern, "replicate": replicate, "role": role,
            "change": st.change, "length": len(items), "cost_per_item": totals["cost"] / len(items),
            "post_windows_below": (float(np.mean([r["theta"] is not None and r["theta"] < A.ALPHA for r in post])) if post else None),
            "first_retrain_after_change": next((r["t"] - st.change for r in windows if r["retrain"] and r["t"] > st.change), None),
            "wall_s": round(time.perf_counter() - t_start, 1), **totals, "windows": windows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--drift", required=True, choices=D.DRIFTS)
    ap.add_argument("--pattern", default="abrupt", choices=("abrupt", "gradual"))
    ap.add_argument("--replicate", type=int, default=0)
    ap.add_argument("--role", type=int, default=0)
    ap.add_argument("--length", type=int, default=D.STREAM_LEN)
    ap.add_argument("--out")
    a = ap.parse_args()
    res = run(a.arm, json.loads(a.cfg), a.drift, a.pattern, a.replicate, a.role, a.length)
    line = json.dumps(res)
    if a.out:
        with open(a.out, "a") as f:
            f.write(line + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "windows"}))


if __name__ == "__main__":
    main()
