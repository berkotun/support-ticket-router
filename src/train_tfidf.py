"""Area B: classical baseline - tf-idf features + logistic regression.

Features: word 1-2 grams + character 2-5 grams (robust to typos), computed on
the preprocessed text (normalization + spelling correction). Logistic
regression over all 151 labels (the "oos" label is a class of its own). The
regularization strength C is chosen on the validation set.

Usage:  python -m src.train_tfidf
"""
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from . import config
from .prepare_data import load_split
from .preprocess import preprocess


def build(C):
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=20_000,
                                 sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                 max_features=30_000, sublinear_tf=True)),
    ])
    return Pipeline([("tfidf", features),
                     ("lr", LogisticRegression(C=C, max_iter=3000))])


def main():
    config.ensure_dirs()
    train, val = load_split("train"), load_split("val")
    Xtr = [preprocess(t) for t in train.text]
    Xva = [preprocess(t) for t in val.text]
    best = None
    for C in (1.0, 5.0, 20.0, 50.0):
        pipe = build(C).fit(Xtr, train.intent)
        acc = float(np.mean(pipe.predict(Xva) == val.intent))
        print(f"  C={C:5}  val accuracy={acc:.4f}")
        if best is None or acc > best[0]:
            best = (acc, C, pipe)
    print(f"  selected C={best[1]}")
    pipe = best[2]
    lr = pipe.named_steps["lr"]
    lr.coef_ = lr.coef_.astype(np.float32)          # store weights as FP32 (half the size)
    lr.intercept_ = lr.intercept_.astype(np.float32)
    joblib.dump(pipe, config.MODELS / "tfidf_lr.joblib", compress=3)


if __name__ == "__main__":
    main()
