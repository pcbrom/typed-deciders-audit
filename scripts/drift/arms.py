"""Arms of the drift experiment (preregistration/drift.md, section 4).

An arm decides three things: which items to accept (a threshold on the top-label confidence, or a
per-bin rule for recalibration), when to retrain, and whether to promote a retrained model. All
arms share the retraining procedure, the audit rate and the label stream (run.py).
"""

import sys
from pathlib import Path

import numpy as np
from scipy.special import betainc, betaincinv, digamma, polygamma
from scipy.stats import beta as _beta
from scipy.stats import binom, t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ablation_gate import beta_cfg, pick, threshold  # noqa: E402

ALPHA = 0.95
LOGIT_MAX, Q_MIN = 15.0, 1e-6                 # numerical guards of the dynamic model (logit of 1 - 3e-7, variance floor)
RULES = {"beta": beta_cfg(0.05, "uniform", "ascending"), "ltt": {"rule": "ltt", "delta": 0.05, "min_n": 20},
         "point": {"rule": "point", "min_n": 20}}
NEVER = 1.01                                 # threshold that accepts nothing
beta_ppf = _beta.ppf


def weighted_beta(conf, ok, w, q, scan, min_n=20):
    """Beta rule on weighted counts rescaled to the effective sample size, (sum w)^2 / sum w^2."""
    order = np.argsort(-conf, kind="stable")
    cs, oks, ws = conf[order], ok[order].astype(float), w[order]
    W, C, W2 = np.cumsum(ws), np.cumsum(ws * oks), np.cumsum(ws ** 2)
    last = np.r_[cs[1:] != cs[:-1], True]
    thr, W, C, W2 = cs[last], W[last], C[last], W2[last]
    n = W ** 2 / W2
    c = C * n / W
    keep = n >= min_n
    thr, n, c = thr[keep], n[keep], c[keep]
    if len(thr) == 0:
        return None
    lower = betaincinv(c + 1, n - c + 1, q)
    testable = betaincinv(n + 1, 1.0, q) >= ALPHA
    return pick(thr, lower >= ALPHA, scan, testable)


def qualify(rule, conf, ok, q=0.05, w=None):
    if w is not None and rule in ("beta", "beta_fixed"):
        tau = weighted_beta(conf, ok, w, q, "ascending" if rule == "beta" else "sequence_testable")
        return NEVER if tau is None else tau
    cfg = dict(RULES[rule]) if rule in RULES else {}
    if rule == "beta":
        cfg = beta_cfg(q, "uniform", "ascending")
    elif rule == "beta_fixed":
        cfg = beta_cfg(q, "uniform", "sequence_testable")
    elif rule == "ltt":
        cfg["delta"] = q
    tau = threshold(cfg, conf, ok, ALPHA)
    return NEVER if tau is None else tau


def promote(test, new_ok, old_ok, q=0.05):
    """Promotion decision on a held-out labelled set, from paired correctness of new and old model."""
    b, c = int((new_ok & ~old_ok).sum()), int((~new_ok & old_ok).sum())
    if test == "always":
        return True
    if test == "beta":                       # P(new better | discordant pairs) under Beta(b+1, c+1)
        return 1.0 - betainc(b + 1, c + 1, 0.5) >= 1 - q
    if test == "mcnemar":                    # exact one-sided McNemar (Dietterich, 1998)
        return b + c > 0 and binom.sf(b - 1, b + c, 0.5) <= q
    if test == "seldonian":                  # one-sided t lower bound on the paired gain (Thomas et al., 2019)
        d = new_ok.astype(float) - old_ok.astype(float)
        n = len(d)
        if n < 2 or d.std(ddof=1) == 0:
            return d.mean() > 0
        return d.mean() - student_t.ppf(1 - q, n - 1) * d.std(ddof=1) / np.sqrt(n) > 0
    raise ValueError(test)


class Arm:
    """Base arm: fixed threshold, never retrains."""
    requal_rule = "ltt"
    promotion = "always"

    def __init__(self, **cfg):
        self.cfg = cfg
        self.q = cfg.get("q", 0.05)
        self.tau = NEVER

    def initial(self, conf, ok):
        self.tau = qualify(self.requal_rule, conf, ok, self.q)

    def accept(self, conf):
        return conf >= self.tau

    def observe(self, conf, ok, accepted, audited, labelled):
        """Called once per window with the window's confidences, correctness and masks."""

    def wants_retrain(self, t):
        return False

    def after_retrain(self, promoted):
        """Reset trigger state after a retraining attempt."""

    def requalify(self, conf, ok, w=None):
        self.tau = qualify(self.requal_rule, conf, ok, self.q, w)

    def per_window_requalify(self):
        return False


class Frozen(Arm):
    pass


class Periodic(Arm):
    def __init__(self, **cfg):
        super().__init__(**cfg)
        self.every, self.promotion, self.last = cfg.get("every", 2000), cfg.get("promotion", "always"), 0

    def wants_retrain(self, t):
        self.now = t
        return t - self.last >= self.every

    def after_retrain(self, promoted):
        self.last = self.now


class Detector(Arm):
    """DDM, ADWIN or Page-Hinkley from river on the audit error stream (or on all labels)."""

    def __init__(self, kind, **cfg):
        super().__init__(**cfg)
        from river import drift
        from river.drift.binary import DDM
        self.kind, self.promotion, self.all_labels = kind, cfg.get("promotion", "always"), cfg.get("all_labels", False)
        params = cfg.get("params", {})
        self.make = {"ddm": lambda: DDM(**params), "adwin": lambda: drift.ADWIN(**params),
                     "ph": lambda: drift.PageHinkley(**params)}[kind]
        self.det, self.flag = self.make(), False

    def observe(self, conf, ok, accepted, audited, labelled):
        mask = labelled if self.all_labels else audited
        for err in (~ok[mask]).astype(int):
            self.det.update(int(err))
            if self.det.drift_detected:
                self.flag = True

    def wants_retrain(self, t):
        return self.flag

    def after_retrain(self, promoted):
        self.det, self.flag = self.make(), False


class LTTWindow(Arm):
    def per_window_requalify(self):
        return True


class ACI(Arm):
    """Threshold adapted online toward the target error rate among accepted items (Gibbs and Candes, 2021)."""

    def __init__(self, **cfg):
        super().__init__(**cfg)
        self.gamma = cfg.get("gamma", 0.01)

    def observe(self, conf, ok, accepted, audited, labelled):
        for err in (~ok[audited]).astype(float):
            self.tau = float(np.clip(self.tau + self.gamma * (err - (1 - ALPHA)), 0.0, NEVER))


class Recalibration(Arm):
    """Laplace succession per confidence bin with forgetting; accept a bin when its estimate reaches the target."""

    def __init__(self, **cfg):
        super().__init__(**cfg)
        self.edges = np.linspace(0, 1, cfg.get("bins", 10) + 1)
        self.lam = cfg.get("lam", 1.0)
        self.s = np.zeros(len(self.edges) - 1)
        self.n = np.zeros(len(self.edges) - 1)

    def _bin(self, conf):
        return np.clip(np.searchsorted(self.edges, conf, side="right") - 1, 0, len(self.s) - 1)

    def initial(self, conf, ok):
        b = self._bin(conf)
        np.add.at(self.s, b, ok.astype(float))
        np.add.at(self.n, b, 1.0)

    def accept(self, conf):
        return ((self.s + 1) / (self.n + 2))[self._bin(conf)] >= ALPHA

    def observe(self, conf, ok, accepted, audited, labelled):
        b = self._bin(conf[labelled])
        self.s *= self.lam
        self.n *= self.lam
        np.add.at(self.s, b, ok[labelled].astype(float))
        np.add.at(self.n, b, 1.0)


class Author(Arm):
    """Beta trigger on audited accepted items with forgetting, Beta promotion, Beta re-qualification.

    Attribution-ablation options (preregistration/drift.md, section 4), each swapping one component:
    counts="window" resets the counts every 500 items instead of discounting by lambda;
    trigger="cp" retrains when the one-sided Clopper-Pearson upper bound of the accuracy is below alpha,
    instead of the Beta posterior with a uniform prior; promotion in {"beta", "mcnemar", "always"};
    requal="ltt" re-qualifies the gate by LTT instead of the Beta rule. The defaults are the arm itself.

    Second-ablation options (implementation_notes.md, repairing the re-qualification): requal="beta_fixed"
    uses the fixed-sequence scan; requal_data="holdout" re-qualifies on the held-out promotion items and
    keeps the previous threshold when none qualifies; requal_weights=True weighs audited items by 1/audit
    rate. The defaults reproduce the arm of the main run.
    """
    requal_rule = "beta"

    def __init__(self, **cfg):
        super().__init__(**cfg)
        self.lam, self.promotion = cfg.get("lam", 0.99), cfg.get("promotion", "beta")
        self.counts, self.trigger = cfg.get("counts", "discount"), cfg.get("trigger", "beta")
        self.requal_rule = cfg.get("requal", "beta")
        self.requal_data, self.requal_weights = cfg.get("requal_data", "recent"), cfg.get("requal_weights", False)
        self.a = self.b = 0.0
        self.seen = 0

    def observe(self, conf, ok, accepted, audited, labelled):
        if self.counts == "window":
            self.seen += len(ok)
            if self.seen >= 500:
                self.a = self.b = 0.0
                self.seen = 0
        lam = 1.0 if self.counts == "window" else self.lam
        for good in ok[audited]:
            self.a, self.b = lam * self.a + float(good), lam * self.b + float(not good)

    def wants_retrain(self, t):
        if self.trigger == "cp":                 # one-sided Clopper-Pearson upper bound of accuracy below alpha
            k, n = self.a, self.a + self.b
            if n < 1:
                return False
            upper = 1.0 if k >= n else float(beta_ppf(1 - self.q, k + 1, n - k))
            return upper < ALPHA
        return betainc(self.a + 1, self.b + 1, ALPHA) >= 1 - self.q   # P(theta < alpha | audit labels)

    def requalify(self, conf, ok, w=None):
        new = qualify(self.requal_rule, conf, ok, self.q, w if self.requal_weights else None)
        if not (self.requal_data == "holdout" and new == NEVER):
            self.tau = new

    def after_retrain(self, promoted):
        self.a = self.b = 0.0
        self.seen = 0


def beta_from_moments(f, q, iters=60):
    """Beta(r, s) whose logit has mean f and variance q, i.e. digamma(r) - digamma(s) = f and trigamma(r) + trigamma(s) = q,
    by Newton's method in (log r, log s) from the approximation r = (1 + e^f)/q, s = (1 + e^-f)/q of West, Harrison and
    Migon (1985), which is accurate only for small q and gives s < 1 and a runaway logit when labels per window are few."""
    f, q = np.asarray(f, float), np.asarray(q, float)
    lr = np.log(np.clip((1 + np.exp(np.clip(f, -30, 30))) / q, 1e-3, 1e8))
    ls = np.log(np.clip((1 + np.exp(-np.clip(f, -30, 30))) / q, 1e-3, 1e8))
    for _ in range(iters):
        r, s = np.exp(lr), np.exp(ls)
        g1, g2 = digamma(r) - digamma(s) - f, polygamma(1, r) + polygamma(1, s) - q
        a, b = polygamma(1, r) * r, -polygamma(1, s) * s
        c, d = polygamma(2, r) * r, polygamma(2, s) * s
        det = a * d - b * c
        step_r, step_s = np.clip((d * g1 - b * g2) / det, -2, 2), np.clip((-c * g1 + a * g2) / det, -2, 2)
        lr, ls = lr - step_r, ls - step_s
        if max(np.max(np.abs(step_r)), np.max(np.abs(step_s))) < 1e-10:
            break
    return np.exp(lr), np.exp(ls)


class Dynamic(Arm):
    """Dynamic generalised linear model with binomial observations and logit link per confidence bin (West, Harrison
    and Migon, 1985), with a level, an optional harmonic seasonal pair of known period (in windows) and optional
    regression on window inputs, evolved by component discount factors (implementation_notes.md, drift with a pattern).
    A bin is accepted when its one-step-ahead predictive mean accuracy reaches alpha, or, with criterion="lower", when
    the q-quantile of its predictive Beta does."""

    def __init__(self, **cfg):
        super().__init__(**cfg)
        self.edges = np.linspace(0, 1, cfg.get("bins", 20) + 1)
        self.season, self.exog = cfg.get("period"), cfg.get("exog", False)
        self.criterion = cfg.get("criterion", "mean")
        disc = [cfg.get("delta", 0.9)]
        F = [1.0]
        G = [np.eye(1)]
        if self.season:
            w = 2 * np.pi / self.season
            G.append(np.array([[np.cos(w), np.sin(w)], [-np.sin(w), np.cos(w)]]))
            F += [1.0, 0.0]
            disc += [0.98, 0.98]
        if self.exog:
            G.append(np.eye(2))
            F += [0.0, 0.0]
            disc += [0.99, 0.99]
        p = len(F)
        self.G = np.zeros((p, p))
        k = 0
        for g in G:
            self.G[k:k + len(g), k:k + len(g)] = g
            k += len(g)
        self.F0, self.disc = np.array(F), np.array(disc)
        nb = len(self.edges) - 1
        self.m, self.C = np.zeros((nb, p)), np.tile(np.eye(p) * 0.5, (nb, 1, 1))
        self.x = np.zeros(2)
        self.ok_bins = np.zeros(nb, bool)

    def _bin(self, conf):
        return np.clip(np.searchsorted(self.edges, conf, side="right") - 1, 0, len(self.m) - 1)

    def _F(self):
        F = self.F0.copy()
        if self.exog:
            F[-2:] = self.x
        return F

    def initial(self, conf, ok):
        b = self._bin(conf)
        s, n = np.zeros(len(self.m)), np.zeros(len(self.m))
        np.add.at(s, b, ok.astype(float))
        np.add.at(n, b, 1.0)
        r, t = s + 1, n - s + 1
        self.m[:, 0] = digamma(r) - digamma(t)
        self.C[:, 0, 0] = polygamma(1, r) + polygamma(1, t)

    def prepare(self, x):
        """Evolve the state to the next window and set the accepted bins from the predictive distribution."""
        self.x = np.asarray(x, float)
        F = self._F()
        a = self.m @ self.G.T
        P = np.einsum("ij,bjk,lk->bil", self.G, self.C, self.G)
        d = np.sqrt(self.disc)
        self.a, self.R = a, P / np.outer(d, d)
        f = np.clip(a @ F, -LOGIT_MAX, LOGIT_MAX)
        q = np.maximum(np.einsum("i,bij,j->b", F, self.R, F), Q_MIN)
        self.f, self.q = f, q
        r, s = beta_from_moments(f, q)
        acc = r / (r + s) if self.criterion == "mean" else beta_ppf(self.q_level, r, s)
        self.ok_bins = acc >= ALPHA
        self.r, self.s = r, s

    q_level = 0.05

    def accept(self, conf):
        return self.ok_bins[self._bin(conf)]

    def observe(self, conf, ok, accepted, audited, labelled):
        b = self._bin(conf[labelled])
        y, n = np.zeros(len(self.m)), np.zeros(len(self.m))
        np.add.at(y, b, ok[labelled].astype(float))
        np.add.at(n, b, 1.0)
        F = self._F()
        m, C = self.a.copy(), self.R.copy()
        for k in np.nonzero(n)[0]:
            r1, s1 = self.r[k] + y[k], self.s[k] + n[k] - y[k]
            f1, q1 = digamma(r1) - digamma(s1), polygamma(1, r1) + polygamma(1, s1)
            RF = self.R[k] @ F
            mk = self.a[k] + RF * (f1 - self.f[k]) / self.q[k]
            Ck = self.R[k] - np.outer(RF, RF) * (1 - min(q1 / self.q[k], 1.0)) / self.q[k]
            Ck = (Ck + Ck.T) / 2 + np.eye(len(F)) * 1e-9
            if np.all(np.isfinite(mk)) and np.all(np.isfinite(Ck)):      # otherwise the bin keeps its prior
                m[k], C[k] = mk, Ck
        self.m, self.C = m, C


def build(name, cfg):
    kinds = {"frozen": Frozen, "periodic": Periodic, "ltt_window": LTTWindow, "aci": ACI, "recal": Recalibration,
             "author": Author, "dynamic": Dynamic}
    if name in ("ddm", "adwin", "ph"):
        return Detector(name, **cfg)
    return kinds[name](**cfg)
