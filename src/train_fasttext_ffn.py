"""Area A: our own fastText embeddings + a feedforward neural network
trained with SGD and backpropagation.

1) fastText (skip-gram with character n-grams, Bojanowski et al., 2017) is
   trained from scratch on the training messages only. Character n-grams give
   vectors even for misspelled or unseen words.
2) Each message = the average of its word vectors (100 dims), standardized.
3) Feedforward network: 100 -> 256 (ReLU, dropout 0.3) -> 151 classes.
   Trained with torch.optim.SGD (momentum 0.9); loss.backward() is
   backpropagation. The best epoch on the validation set is kept.

Usage:  python -m src.train_fasttext_ffn
"""

import numpy as np
import torch
import torch.nn as nn
from gensim.models import FastText

from . import config
from .prepare_data import load_split
from .preprocess import preprocess

DIM = 100


class FFN(nn.Module):
    def __init__(self, n_classes, d_in=DIM, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)          # logits; softmax is applied in the loss / at prediction


def embed(texts, kv) -> np.ndarray:
    """Message vector = mean of its word vectors (fastText also covers unseen words)."""
    out = np.zeros((len(texts), kv.vector_size), dtype=np.float32)
    for i, t in enumerate(texts):
        words = t.split()
        if words:
            out[i] = np.mean([kv[w] for w in words], axis=0)
    return out


def main(epochs=60, lr=0.05, batch=64):
    config.ensure_dirs()
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    train, val = load_split("train"), load_split("val")
    labels = sorted(set(train.intent))
    lid = {l: i for i, l in enumerate(labels)}
    tr_txt = [preprocess(t) for t in train.text]
    va_txt = [preprocess(t) for t in val.text]

    print("[1/2] Training fastText (skip-gram + char n-grams, training messages only)...")
    ft = FastText([t.split() for t in tr_txt], vector_size=DIM, window=5, min_count=1,
                  sg=1, epochs=20, min_n=3, max_n=5, bucket=100_000, workers=1,
                  seed=config.SEED)
    ft.wv.save(str(config.MODELS / "fasttext.kv"))
    print(f"  vocabulary: {len(ft.wv):,} words")

    Xtr, Xva = torch.tensor(embed(tr_txt, ft.wv)), torch.tensor(embed(va_txt, ft.wv))
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6       # standardization helps SGD a lot
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    ytr = torch.tensor([lid[l] for l in train.intent])
    yva = np.array([lid[l] for l in val.intent])

    print("[2/2] Feedforward network (SGD + backpropagation)")
    model = FFN(len(labels))
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    best_acc, best_state = -1, None
    for ep in range(1, epochs + 1):
        model.train()
        perm, total = torch.randperm(len(Xtr)), 0.0
        for i in range(0, len(Xtr), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])   # forward pass
            loss.backward()                              # backpropagation
            opt.step()                                   # SGD update
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            acc = float(np.mean(model(Xva).argmax(1).numpy() == yva))
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == 1:
            print(f"  epoch {ep:2d}  train loss={total / len(Xtr):.4f}  val accuracy={acc:.4f}")
    print(f"  best val accuracy={best_acc:.4f}")
    torch.save({"state": best_state, "mu": mu, "sd": sd, "labels": labels},
               config.MODELS / "ffn.pt")


if __name__ == "__main__":
    main()
