"""Natural noise operators for utterances (implementation_notes.md, robustness to natural noise).

Every operator is deterministic given its seed. Conditions are named "<operator>" or "<operator>_<rate>".
"""

import re
import unicodedata

import numpy as np

ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")
NEIGHBOURS = {}
for r, row in enumerate(ROWS):
    for c, ch in enumerate(row):
        near = set()
        for dr in (-1, 0, 1):
            rr = r + dr
            if 0 <= rr < len(ROWS):
                for dc in (-1, 0, 1):
                    cc = c + dc
                    if 0 <= cc < len(ROWS[rr]) and (dr, dc) != (0, 0):
                        near.add(ROWS[rr][cc])
        NEIGHBOURS[ch] = sorted(near)
OPS = ("typo", "ordem", "caixa", "acento", "misto")
ERRORS = ("substituicao", "omissao", "duplicacao", "transposicao")


def strip_accents(s):
    return "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")


def base(ch):
    return strip_accents(ch).lower()


def keyboard_error(word, kind, rng):
    """One keyboard error inside a word of at least two letters."""
    letters = [i for i, ch in enumerate(word) if ch.isalpha()]
    if len(letters) < 2:
        return word
    i = int(letters[rng.integers(len(letters))])
    if kind == "substituicao":
        near = NEIGHBOURS.get(base(word[i]))
        if not near:
            return word
        ch = near[int(rng.integers(len(near)))]
        return word[:i] + (ch.upper() if word[i].isupper() else ch) + word[i + 1:]
    if kind == "omissao":
        return word[:i] + word[i + 1:]
    if kind == "duplicacao":
        return word[:i] + word[i] + word[i:]
    j = i + 1 if i + 1 < len(word) else i - 1        # transposition with the next letter, or the previous one at the end
    a, b = sorted((i, j))
    return word[:a] + word[b] + word[a] + word[b + 1:]


def typo(text, rate, rng):
    words = text.split()
    eligible = [k for k, w in enumerate(words) if sum(ch.isalpha() for ch in w) >= 2]
    if not eligible or rate <= 0:
        return text
    hit = [k for k in eligible if rng.random() < rate]
    if not hit:
        hit = [eligible[int(rng.integers(len(eligible)))]]
    for k in hit:
        words[k] = keyboard_error(words[k], ERRORS[int(rng.integers(len(ERRORS)))], rng)
    return " ".join(words)


def ordem(text, rate, rng):
    words = text.split()
    k = 0
    while k < len(words) - 1:
        if rng.random() < rate:
            words[k], words[k + 1] = words[k + 1], words[k]
            k += 2
        else:
            k += 1
    return " ".join(words)


def caixa(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


def misto(text, rate, rng):
    words = caixa(text).split()
    words = [strip_accents(w) if rng.random() < 0.5 else w for w in words]
    return typo(" ".join(words), rate, rng)


def apply(text, condition, item, op_index=None):
    """Apply a named condition, e.g. 'typo_0.10', 'caixa', 'misto_0.20', to one utterance."""
    if condition == "limpo":
        return text
    op, _, r = condition.partition("_")
    rate = float(r) if r else 0.0
    rng = np.random.default_rng(np.random.SeedSequence([20261023, 10, OPS.index(op), int(round(rate * 100)), item]))
    if op == "typo":
        return typo(text, rate, rng)
    if op == "ordem":
        return ordem(text, rate, rng)
    if op == "caixa":
        return caixa(text)
    if op == "acento":
        return strip_accents(text)
    return misto(text, rate, rng)


def conditions(loc):
    out = ["limpo", "typo_0.05", "typo_0.10", "typo_0.20", "ordem_0.10", "caixa", "misto_0.05", "misto_0.10", "misto_0.20"]
    return out + (["acento"] if loc == "pt" else [])
