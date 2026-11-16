"""Original contribution: a "typo" robustness test set.

Real customers type fast on phones and make typos. This module copies the
OFFICIAL test set and adds realistic keyboard typos with a fixed random seed
(42). Labels do not change, only the spelling. Every model is evaluated on
both the clean and the typo test set, which shows how much the edit-distance
spelling correction helps.

Typo types (one per word, applied to ~35 % of the words with 4+ letters):
    neighbor   : a key next to it on a QWERTY keyboard   "balance" -> "balancr"
    delete     : drop one letter                         "balance" -> "balnce"
    transpose  : swap two neighbouring letters           "balance" -> "blaance"
    duplicate  : type one letter twice                   "balance" -> "ballance"
"""
import random

import pandas as pd

ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
NEIGHBORS = {}
for r, row in enumerate(ROWS):
    for c, ch in enumerate(row):
        near = [row[c - 1] if c > 0 else "", row[c + 1] if c + 1 < len(row) else ""]
        for rr in (r - 1, r + 1):
            if 0 <= rr < len(ROWS) and c < len(ROWS[rr]):
                near.append(ROWS[rr][c])
        NEIGHBORS[ch] = [n for n in near if n]


def _neighbor(w, rng):
    i = rng.randrange(len(w))
    if w[i] not in NEIGHBORS:
        return w
    return w[:i] + rng.choice(NEIGHBORS[w[i]]) + w[i + 1:]


def _delete(w, rng):
    i = rng.randrange(1, len(w))
    return w[:i] + w[i + 1:]


def _transpose(w, rng):
    i = rng.randrange(1, len(w) - 1)
    return w[:i] + w[i + 1] + w[i] + w[i + 2:]


def _duplicate(w, rng):
    i = rng.randrange(len(w))
    return w[:i] + w[i] + w[i:]


TYPOS = [_neighbor, _delete, _transpose, _duplicate]


def add_typos(text: str, rng: random.Random, rate: float = 0.35) -> str:
    out = []
    for tok in text.split(" "):
        if len(tok) >= 4 and tok.isalpha() and rng.random() < rate:
            tok = rng.choice(TYPOS)(tok, rng)
        out.append(tok)
    return " ".join(out)


def make_typo_set(test: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    noisy = test.copy()
    noisy["text"] = [add_typos(t, rng) for t in noisy.text]
    return noisy


if __name__ == "__main__":
    rng = random.Random(1)
    for s in ["what is my current account balance", "please book a flight to boston"]:
        print(s, "->", add_typos(s, rng, rate=1.0))
