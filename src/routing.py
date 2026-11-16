"""Turning model probabilities into a routing decision.

1) The predicted intent is the label with the highest probability.
2) If that label is "oos" or its probability is below the model's threshold
   tau, the message is treated as out-of-scope -> "Human agent".
   tau is chosen per model on the VALIDATION set (benchmark.py) to maximize
   the mean of in-scope accuracy and out-of-scope recall.
3) Small talk is answered with an ELIZA-style rule-based reply instead of
   being routed: when an ELIZA rule matches and the model sees small talk or
   out-of-scope, or when the model predicts a small_talk intent.
"""
import json

import numpy as np

from . import config
from .domains import HUMAN, domain_of, team_of
from .eliza import reply, reply_for_intent

THRESHOLDS_FILE = "thresholds.json"


def final_intents(probs: np.ndarray, labels: list, tau: float) -> list:
    top = probs.argmax(1)
    conf = probs.max(1)
    return [config.OOS if labels[i] == config.OOS or c < tau else labels[i]
            for i, c in zip(top, conf)]


def load_thresholds() -> dict:
    path = config.MODELS / THRESHOLDS_FILE
    return json.loads(path.read_text()) if path.exists() else {}


def decide(text: str, p: np.ndarray, labels: list, tau: float) -> dict:
    """Routing decision for one message and one model's probabilities p."""
    i = int(p.argmax())
    intent, conf = labels[i], float(p[i])
    final = final_intents(p[None, :], labels, tau)[0]
    domain = domain_of(final)
    rule = reply(text)
    if rule and domain in ("small_talk", "oos"):
        return {"intent": intent, "conf": conf, "final": final, "team": "Auto-reply (ELIZA rules)",
                "reply": rule}
    if domain == "small_talk":
        return {"intent": intent, "conf": conf, "final": final, "team": team_of(final),
                "reply": reply_for_intent(final)}
    return {"intent": intent, "conf": conf, "final": final,
            "team": HUMAN if final == config.OOS else team_of(final), "reply": None}
