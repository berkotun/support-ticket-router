"""Shared code for the Transformer models (Area C) and distillation (Area D).

finetune() trains a pretrained encoder with a classification head on the 151
labels. Without teacher logits it is ordinary fine-tuning (cross-entropy).
With teacher logits it is knowledge distillation (Hinton et al., 2015):

    loss = alpha * T^2 * KL( softmax(teacher / T) || softmax(student / T) )
         + (1 - alpha) * cross_entropy(student, true label)

The T^2 factor keeps the gradient scale of the soft loss comparable to the
hard loss. The checkpoint with the best validation accuracy is saved.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from . import config
from .preprocess import preprocess


def encode(tok, texts):
    """Preprocess (normalize + spell-correct), then WordPiece-tokenize."""
    return tok([preprocess(t) for t in texts], truncation=True, max_length=config.MAX_LEN,
               padding=True, return_tensors="pt")


@torch.no_grad()
def predict_logits(model, tok, texts, device="cpu", batch=64):
    model.eval()
    out = []
    for i in range(0, len(texts), batch):
        enc = encode(tok, list(texts[i:i + batch])).to(device)
        out.append(model(**enc).logits.float().cpu())
    return torch.cat(out) if out else torch.zeros(0)


def finetune(model_name, out_dir, train, val, labels, teacher_logits=None):
    torch.manual_seed(config.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| model:", model_name)
    lid = {l: i for i, l in enumerate(labels)}
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels),
        id2label=dict(enumerate(labels)), label2id=lid,
        ignore_mismatched_sizes=True).to(device)      # a new 151-way classification head

    y = [lid[l] for l in train.intent]
    idx = list(range(len(train)))
    texts = list(train.text)

    def collate(batch_idx):
        enc = encode(tok, [texts[i] for i in batch_idx])
        return enc, torch.tensor([y[i] for i in batch_idx]), torch.tensor(batch_idx)

    loader = DataLoader(idx, batch_size=config.BATCH, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(config.SEED))
    opt = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=0.01)
    steps = len(loader) * config.EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    use_amp = device == "cuda"              # mixed precision on GPU for speed
    scaler = torch.amp.GradScaler(enabled=use_amp)
    T, alpha = config.KD_TEMPERATURE, config.KD_ALPHA
    yva = np.array([lid[l] for l in val.intent])

    best = -1.0
    for ep in range(1, config.EPOCHS + 1):
        model.train()
        total = 0.0
        for enc, yb, ib in loader:
            enc, yb = enc.to(device), yb.to(device)
            with torch.autocast(device_type=device, enabled=use_amp):
                logits = model(**enc).logits.float()
            loss = F.cross_entropy(logits, yb)
            if teacher_logits is not None:              # knowledge distillation
                t = teacher_logits[ib].to(device)
                soft = F.kl_div(F.log_softmax(logits / T, -1), F.softmax(t / T, -1),
                                reduction="batchmean") * T * T
                loss = alpha * soft + (1 - alpha) * loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        acc = float(np.mean(predict_logits(model, tok, list(val.text), device).argmax(1).numpy() == yva))
        print(f"epoch {ep}: train loss={total / len(loader):.4f}  val accuracy={acc:.4f}")
        if acc > best:
            best = acc
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
            print("  -> best model saved to", out_dir)
    print(f"best val accuracy={best:.4f}")
    return model.to("cpu").eval(), tok
