"""Exact coalition games: Möbius transform, Shapley values, first- and last-entry evidence, pairwise
Shapley interaction indices and the share of variance explained up to order two.

A game on d units is an array v of length 2^d indexed by the bit mask of the coalition. All
transforms are exact and cost O(d 2^d).
"""

import numpy as np
from scipy.special import logit as _logit

EPS = 1e-6


def value(probs, j, k=None):
    """y(S) = logit p_j(x_S), or log p_j - log p_k when k is given; probabilities clipped to [EPS, 1-EPS]."""
    p = np.clip(np.asarray(probs, float), EPS, 1 - EPS)
    return np.log(p[:, j]) - np.log(p[:, k]) if k is not None else _logit(p[:, j])


def mobius(v):
    """m_T = sum over U subset of T of (-1)^{|T|-|U|} v(U), by the fast subset transform."""
    m = np.array(v, float)
    n = m.size
    d = n.bit_length() - 1
    for i in range(d):
        bit = 1 << i
        idx = np.arange(n)
        sel = (idx & bit) != 0
        m[sel] -= m[idx[sel] ^ bit]
    return m


def popcount(n):
    return np.array([bin(x).count("1") for x in range(n)])


def shapley(m, d):
    """E_i = sum over T containing i of m_T / |T|."""
    size = popcount(1 << d)
    out = np.zeros(d)
    for i in range(d):
        has = (np.arange(1 << d) >> i) & 1 == 1
        out[i] = (m[has] / size[has]).sum()
    return out


def first_last(v, d):
    """F_i = v({i}) - v(empty) and W_i = v(N) - v(N without i)."""
    full = (1 << d) - 1
    return np.array([v[1 << i] - v[0] for i in range(d)]), np.array([v[full] - v[full ^ (1 << i)] for i in range(d)])


def interaction(m, d):
    """Pairwise Shapley interaction index I_ij = sum over T containing i and j of m_T / (|T| - 1)."""
    size = popcount(1 << d)
    idx = np.arange(1 << d)
    out = np.zeros((d, d))
    for i in range(d):
        for k in range(i + 1, d):
            has = ((idx >> i) & 1 == 1) & ((idx >> k) & 1 == 1)
            out[i, k] = out[k, i] = (m[has] / (size[has] - 1)).sum()
    return out


def r2_order2(v, m, d):
    """Share of the variance of v over all 2^d coalitions (equal weights) explained by the Möbius terms
    of order at most two."""
    size = popcount(1 << d)
    m2 = np.where(size <= 2, m, 0.0)
    v2 = mobius_inverse(m2)
    ss = ((v - v.mean()) ** 2).sum()
    return 1.0 - ((v - v2) ** 2).sum() / ss if ss > 0 else np.nan


def mobius_inverse(m):
    """v(S) = sum over T subset of S of m_T (zeta transform)."""
    v = np.array(m, float)
    n = v.size
    d = n.bit_length() - 1
    for i in range(d):
        bit = 1 << i
        idx = np.arange(n)
        sel = (idx & bit) != 0
        v[sel] += v[idx[sel] ^ bit]
    return v


def r2_order2_ls(v, d):
    """Share of the variance of v over all 2^d coalitions (equal weights) explained by its least-squares
    projection on the constant, the unit indicators and their pairwise products (G1, see
    preregistration/implementation_notes.md)."""
    idx = np.arange(1 << d)
    z = ((idx[:, None] >> np.arange(d)) & 1).astype(float)
    cols = [np.ones(len(idx))] + [z[:, i] for i in range(d)] + [z[:, i] * z[:, k] for i in range(d) for k in range(i + 1, d)]
    X = np.stack(cols, 1)
    beta, *_ = np.linalg.lstsq(X, v, rcond=None)
    ss = ((v - v.mean()) ** 2).sum()
    return 1.0 - ((v - X @ beta) ** 2).sum() / ss if ss > 0 else np.nan
