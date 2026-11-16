"""Area D (step 2): dynamic INT8 quantization of the Transformer models.

Dynamic quantization (PyTorch quantize_dynamic):
  - The weights of every nn.Linear layer go from float32 (4 bytes) to int8 (1 byte).
  - Activations are scaled and converted to int8 on the fly, per input.
  - No extra training or calibration data is needed; it speeds up CPU inference.
  - Embedding layers stay float32 (they are not Linear layers), so the files
    do not shrink by exactly 4x.

The distilled student is quantized for the app. The teacher is quantized too,
so the results table can separate the effect of distillation from the effect
of quantization.

Usage:   python -m src.quantize
Output:  models/student_int8/, models/teacher_int8/  (config + tokenizer + model_int8.pt)
"""
import shutil

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification

from . import config

INT8_FILE = "model_int8.pt"


def quantize(model):
    return torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def load_int8(folder):
    """Load a saved INT8 model: build the empty architecture -> quantize it ->
    load the int8 weights. The float32 weights are never needed."""
    cfg = AutoConfig.from_pretrained(folder)
    model = quantize(AutoModelForSequenceClassification.from_config(cfg).eval())
    model.load_state_dict(torch.load(folder / INT8_FILE, weights_only=False))
    return model.eval()


def _mb(folder):
    return sum(f.stat().st_size for f in folder.iterdir()) / 2**20


def quantize_folder(name):
    src, dst = config.MODELS / f"{name}_fp32", config.MODELS / f"{name}_int8"
    dst.mkdir(parents=True, exist_ok=True)
    model = AutoModelForSequenceClassification.from_pretrained(src).eval()
    torch.save(quantize(model).state_dict(), dst / INT8_FILE)
    for f in src.iterdir():                     # config + tokenizer, not the FP32 weights
        if f.suffix not in (".safetensors", ".bin"):
            shutil.copy(f, dst / f.name)
    print(f"{name}: FP32 {_mb(src):.1f} MB  ->  INT8 {_mb(dst):.1f} MB")


def main():
    for name in ("student", "teacher"):
        quantize_folder(name)


if __name__ == "__main__":
    main()
