"""Download CLINC150 ("plus" version), save the official splits and build the
spelling-correction vocabulary.

Outputs:
    data/processed/train.csv      15,250 messages (15,000 in-scope + 250 out-of-scope)
    data/processed/val.csv         3,100 messages -> model selection, thresholds
    data/processed/test.csv        5,500 messages (4,500 in-scope + 1,000 out-of-scope)
                                   -> ALL models are evaluated here
    data/processed/test_typos.csv  the same test set with keyboard typos (typos.py)
    models/labels.json             the 151 labels (150 intents + "oos")
    models/spell_vocab.json        vocabulary for spelling correction (training data only)

Usage:  python -m src.prepare_data
"""
import json
import urllib.request

import pandas as pd

from . import config
from .domains import check_labels
from .preprocess import SPELL_FILE
from .spell import SpellCorrector
from .typos import make_typo_set


def _from_huggingface():
    from datasets import load_dataset
    ds = load_dataset(config.HF_DATASET, config.HF_CONFIG)
    names = ds["train"].features["intent"].names
    out = {}
    for split, key in (("train", "train"), ("validation", "val"), ("test", "test")):
        df = ds[split].to_pandas()
        out[key] = pd.DataFrame({"text": df.text, "intent": [names[i] for i in df.intent]})
    return out


def _from_github():
    """Fallback: the original JSON file of the CLINC150 authors."""
    path = config.DATA_RAW / "data_oos_plus.json"
    if not path.exists():
        urllib.request.urlretrieve(config.GITHUB_JSON, path)
    raw = json.loads(path.read_text())
    out = {}
    for key in ("train", "val", "test"):
        rows = raw[key] + raw[f"oos_{key}"]
        out[key] = pd.DataFrame(rows, columns=["text", "intent"])
    return out


def load_raw():
    local = config.DATA_RAW / "data_oos_plus.json"
    if local.exists():
        return _from_github()
    try:
        return _from_huggingface()
    except Exception as e:  # noqa: BLE001
        print("Hugging Face download failed, trying GitHub:", e)
        return _from_github()


def main():
    config.ensure_dirs()
    splits = load_raw()
    splits["test_typos"] = make_typo_set(splits["test"], seed=config.SEED)
    for name, df in splits.items():
        df.to_csv(config.DATA_PROC / f"{name}.csv", index=False)
        n_oos = (df.intent == config.OOS).sum()
        print(f"{name:10} {len(df):6d} messages | in-scope {len(df) - n_oos:5d} | out-of-scope {n_oos:5d}")

    labels = sorted(set(splits["train"].intent))
    check_labels(labels)
    (config.MODELS / "labels.json").write_text(json.dumps(labels))
    print(f"labels: {len(labels)}")

    sc = SpellCorrector.from_texts(splits["train"].text)
    sc.save(config.MODELS / SPELL_FILE)
    print(f"spelling vocabulary: {len(sc.freq):,} words")


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(config.DATA_PROC / f"{name}.csv", keep_default_na=False)


def load_labels() -> list:
    return json.loads((config.MODELS / "labels.json").read_text())


if __name__ == "__main__":
    main()
