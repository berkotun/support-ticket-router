"""Area D (step 1): knowledge distillation into a smaller student model.

Teacher: our fine-tuned BERT-base (12 layers, 110M parameters).
Student: DistilBERT (6 layers, 66M parameters), initialised from the
pretrained distilbert-base-uncased checkpoint.

The teacher's logits on the training messages are computed once; the student
then learns from both the teacher's soft probabilities (temperature T = 2)
and the true labels (see transformer_utils.finetune).

Usage:   python -m src.distill
Output:  models/student_fp32/
"""
import torch

from . import config
from .prepare_data import load_labels, load_split
from .transformer_utils import finetune, predict_logits


def main():
    config.ensure_dirs()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    train, val, labels = load_split("train"), load_split("val"), load_labels()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tdir = config.MODELS / "teacher_fp32"
    teacher = AutoModelForSequenceClassification.from_pretrained(tdir).to(device)
    tok = AutoTokenizer.from_pretrained(tdir)
    print("computing teacher logits on the training set...")
    t_logits = predict_logits(teacher, tok, list(train.text), device)
    del teacher
    finetune(config.STUDENT_NAME, config.MODELS / "student_fp32", train, val, labels,
             teacher_logits=t_logits)


if __name__ == "__main__":
    main()
