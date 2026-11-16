"""Area C: fine-tune BERT (bert-base-uncased, 110M parameters) for intent
classification. A classification head ([CLS] -> dropout -> Linear -> 151
logits) is added on top of the pretrained encoder; the tokenizer is WordPiece.
This model is also the TEACHER for knowledge distillation (distill.py).

Uses the GPU if available (~5-8 min on a Colab T4).

Usage:   python -m src.train_teacher
Output:  models/teacher_fp32/
"""
from . import config
from .prepare_data import load_labels, load_split
from .transformer_utils import finetune


def main():
    config.ensure_dirs()
    finetune(config.TEACHER_NAME, config.MODELS / "teacher_fp32",
             load_split("train"), load_split("val"), load_labels())


if __name__ == "__main__":
    main()
