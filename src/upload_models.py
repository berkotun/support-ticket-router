"""Upload the trained models to the Hugging Face Hub (R7: no large weight files in the repo).

Only what the app needs is uploaded (tf-idf, fastText + FFN, BERT teacher FP32,
DistilBERT student INT8, labels, thresholds, spelling vocabulary). The other
benchmark-only checkpoints (student FP32, teacher INT8) are skipped.

Usage:  huggingface-cli login      (or notebook_login() in Colab)
        python -m src.upload_models YOUR_HF_USERNAME/support-ticket-router-models
"""
import sys

from huggingface_hub import HfApi

from . import config


def main(repo_id):
    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)        # public by default
    api.upload_folder(folder_path=str(config.MODELS), repo_id=repo_id,
                      ignore_patterns=["student_fp32/*", "teacher_int8/*"])
    print(f"Uploaded: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else config.MODELS_REPO)
