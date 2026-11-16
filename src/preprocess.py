"""The shared preprocessing pipeline used by every model, in training, in the
benchmark and in the app:

    raw message -> regex normalization -> spelling correction -> model input

Codes are extracted from the raw message for display in the app.
"""
from . import config
from .normalize import extract_codes, normalize
from .spell import SpellCorrector

SPELL_FILE = "spell_vocab.json"
_SPELL = None


def get_corrector():
    global _SPELL
    if _SPELL is None:
        _SPELL = SpellCorrector.load(config.MODELS / SPELL_FILE)
    return _SPELL


def preprocess(text: str, correct: bool = True) -> str:
    t = normalize(text)
    if correct:
        t, _ = get_corrector().correct(t)
    return t


def analyze(text: str) -> dict:
    """Every preprocessing step, for display in the app."""
    norm = normalize(text)
    corrected, fixes = get_corrector().correct(norm)
    return {"normalized": norm, "corrected": corrected, "fixes": fixes,
            "codes": extract_codes(text)}
