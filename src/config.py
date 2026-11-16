"""Project-wide settings and file paths.

Every script reads its paths and hyper-parameters from here, so a value only
has to be changed in one place.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"            # downloaded raw files
DATA_PROC = ROOT / "data" / "processed"     # train / val / test CSV files
MODELS = Path(os.environ.get("MODELS_DIR", ROOT / "models"))
RESULTS = ROOT / "results"

SEED = 42

# Dataset: CLINC150, "plus" version (Larson et al., 2019), English
HF_DATASET = "clinc/clinc_oos"
HF_CONFIG = "plus"
GITHUB_JSON = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_oos_plus.json"
OOS = "oos"                                  # the out-of-scope label

# Transformers. Both can be pointed to small local models through environment
# variables for quick smoke tests.
TEACHER_NAME = os.environ.get("TEACHER_NAME", "bert-base-uncased")
STUDENT_NAME = os.environ.get("STUDENT_NAME", "distilbert-base-uncased")
MAX_LEN = 48                # CLINC150 messages are short (max ~30 words)
EPOCHS = int(os.environ.get("EPOCHS", 5))
BATCH = 32
LR = float(os.environ.get("LR", 5e-5))

# Knowledge distillation (Hinton et al., 2015)
KD_TEMPERATURE = 2.0
KD_ALPHA = 0.5              # weight of the soft (teacher) loss vs. the hard-label loss

# If the models are not found locally, the app downloads them from this
# Hugging Face Hub repository. Replace with your own user name (see README).
MODELS_REPO = os.environ.get("MODELS_REPO", "berktn/support-ticket-router-models")


def ensure_dirs():
    for p in (DATA_RAW, DATA_PROC, MODELS, RESULTS):
        p.mkdir(parents=True, exist_ok=True)
