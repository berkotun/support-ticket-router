"""Area B: spelling correction with minimum edit distance.

Customers misspell product and service names ("mastercrad", "netflx",
"dishwaser"). Before classification, every unknown word is replaced by the
closest known word:

  vocabulary = words of the training messages (with their frequencies)
               + a product / service catalog (PRODUCTS)
  candidate  = known word with the smallest Levenshtein distance
               (max 1 edit for words of 4-5 letters, max 2 edits for 6+ letters);
               ties are broken by word frequency.

Speed: comparing an unknown word with every vocabulary word is slow, so we
use a symmetric-delete index (the idea behind SymSpell): every word is stored
under all strings obtained by deleting up to 2 of its letters. Two words
within edit distance 2 share at least one such "delete" string, so only a few
candidates have to be checked with the real edit_distance().
"""
import json
from collections import Counter
from functools import lru_cache
from itertools import combinations

from .normalize import normalize

# Product / service names of the NovaAssist catalog (frequent in support messages)
PRODUCTS = [
    "mastercard", "visa", "amex", "discover", "paypal", "venmo", "zelle",
    "checking", "savings", "mortgage", "overdraft", "routing", "statement",
    "netflix", "spotify", "pandora", "alexa", "thermostat", "dishwasher",
    "refrigerator", "microwave", "uber", "lyft", "toyota", "honda", "tesla",
    "delta", "united", "southwest", "marriott", "hilton", "airbnb", "expedia",
    "passport", "luggage", "itinerary", "paycheck", "401k", "w2",
    "insurance", "calendar", "playlist", "reservation", "restaurant",
]
MAX_DELETES = 2


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance by dynamic programming (Jurafsky & Martin, Ch. 2):
    insertion = deletion = substitution = 1."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1,                # deletion
                         cur[j - 1] + 1,             # insertion
                         prev[j - 1] + (ca != cb))   # substitution
        prev = cur
    return prev[-1]


def _deletes(word: str, k: int = MAX_DELETES):
    out = {word}
    for n in range(1, min(k, len(word) - 1) + 1):
        for idx in combinations(range(len(word)), n):
            out.add("".join(c for i, c in enumerate(word) if i not in idx))
    return out


class SpellCorrector:
    def __init__(self, freq: dict):
        self.freq = dict(freq)
        self.index = {}
        for w in self.freq:
            if len(w) >= 4:
                for d in _deletes(w):
                    self.index.setdefault(d, []).append(w)
        self.correct_word = lru_cache(maxsize=100_000)(self._correct_word)

    @classmethod
    def from_texts(cls, texts, min_count=2):
        c = Counter(w for t in texts for w in normalize(t).split())
        freq = {w: n for w, n in c.items() if n >= min_count and w.isalpha()}
        for p in PRODUCTS:                       # catalog words are always known
            freq[p] = max(freq.get(p, 0), 50)
        return cls(freq)

    def _correct_word(self, w: str) -> str:
        if w in self.freq or len(w) < 4 or not w.isalpha():
            return w
        max_d = 1 if len(w) <= 5 else 2
        cands = {c for d in _deletes(w, max_d) for c in self.index.get(d, [])}
        best = None
        for c in cands:
            dist = edit_distance(w, c)
            if dist <= max_d:
                key = (dist, -self.freq[c])
                if best is None or key < best[0]:
                    best = (key, c)
        return best[1] if best else w

    def correct(self, text: str):
        """Returns (corrected text, [(wrong, corrected), ...])."""
        words, fixes = [], []
        for w in text.split():
            core = w.strip("!?.,")
            fixed = self.correct_word(core)
            if fixed != core:
                fixes.append((core, fixed))
                w = w.replace(core, fixed)
            words.append(w)
        return " ".join(words), fixes

    def save(self, path):
        path.write_text(json.dumps(self.freq))

    @classmethod
    def load(cls, path):
        return cls(json.loads(path.read_text()))
