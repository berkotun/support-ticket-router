"""A common interface for all models. Both the benchmark and the app use it,
so the numbers in the results table and the behaviour of the app come from
exactly the same code.

Every model has:
    name, precision, files     -> for the results table
    probs(texts) -> np.ndarray  shape (n_messages, 151), columns ordered like labels.json
"""
from pathlib import Path

import joblib
import numpy as np
import torch
from gensim.models import KeyedVectors
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from . import config
from .prepare_data import load_labels
from .preprocess import preprocess
from .quantize import load_int8
from .train_fasttext_ffn import FFN, embed
from .transformer_utils import predict_logits


def _size_mb(paths):
    total = 0
    for p in map(Path, paths):
        if p.is_dir():
            total += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        elif p.exists():
            total += p.stat().st_size
    return total / 2**20


class Base:
    key = name = precision = ""
    files = []

    def size_mb(self):
        return _size_mb(self.files)

    def probs(self, texts):
        raise NotImplementedError


class TfidfLR(Base):
    precision = "FP32"

    def __init__(self, labels, m=config.MODELS, correct=True):
        self.key = "tfidf" if correct else "tfidf_nospell"
        self.name = "tf-idf + LogReg (classical baseline)" if correct else \
                    "tf-idf + LogReg, no spell correction (ablation)"
        self.correct = correct
        self.files = [m / "tfidf_lr.joblib"]
        self.pipe = joblib.load(self.files[0])
        classes = list(self.pipe.classes_)
        self.order = [classes.index(l) for l in labels]   # align columns with labels.json

    def probs(self, texts):
        x = [preprocess(t, correct=self.correct) for t in texts]
        return self.pipe.predict_proba(x)[:, self.order]


class FastTextFFN(Base):
    key, name, precision = "ffn", "fastText avg + FFN (own neural net)", "FP32"

    def __init__(self, labels, m=config.MODELS):
        self.files = [f for f in m.glob("fasttext.kv*")] + [m / "ffn.pt"]
        self.kv = KeyedVectors.load(str(m / "fasttext.kv"))
        ckpt = torch.load(m / "ffn.pt")
        assert ckpt["labels"] == labels, "label order mismatch"
        self.mu, self.sd = ckpt["mu"], ckpt["sd"]
        self.net = FFN(len(labels))
        self.net.load_state_dict(ckpt["state"])
        self.net.eval()

    @torch.no_grad()
    def probs(self, texts):
        x = torch.tensor(embed([preprocess(t) for t in texts], self.kv))
        return torch.softmax(self.net((x - self.mu) / self.sd), -1).numpy()


class Transformer(Base):
    NAMES = {"teacher": "BERT-base fine-tuned (Transformer, teacher)",
             "student": "DistilBERT distilled (student)"}

    def __init__(self, which, int8, m=config.MODELS):
        d = m / f"{which}_{'int8' if int8 else 'fp32'}"
        self.key = f"{which}_{'int8' if int8 else 'fp32'}"
        self.name = self.NAMES[which] + (" + INT8 quantized" if int8 else "")
        self.precision = "INT8 dynamic" if int8 else "FP32"
        self.files = [d]
        self.tok = AutoTokenizer.from_pretrained(d)
        self.model = load_int8(d) if int8 else \
            AutoModelForSequenceClassification.from_pretrained(d).eval()

    def probs(self, texts):
        return torch.softmax(predict_logits(self.model, self.tok, list(texts)), -1).numpy()


def load_all(m=config.MODELS, for_app=False):
    """Key -> model. The app loads only the four models it shows side by side:
    baseline, own neural net, Transformer (teacher FP32) and the quantized
    distilled Transformer (student INT8), which makes the routing decision."""
    labels = load_labels()
    models = {"tfidf": TfidfLR(labels, m), "ffn": FastTextFFN(labels, m),
              "teacher_fp32": Transformer("teacher", False, m)}
    if not for_app:
        models["teacher_int8"] = Transformer("teacher", True, m)
        models["student_fp32"] = Transformer("student", False, m)
    models["student_int8"] = Transformer("student", True, m)
    if not for_app:
        models["tfidf_nospell"] = TfidfLR(labels, m, correct=False)
    return labels, models
