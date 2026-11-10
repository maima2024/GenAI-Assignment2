"""Upload the LoRA adapters to the Hugging Face Hub (needs HF_TOKEN with write access).

    HF_TOKEN=hf_xxx python scripts/push_to_hub.py --user <hf-username>
"""
import argparse
import os

from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("--user", required=True)
ap.add_argument("--target_dir", default="adapters_export/target_qwen2.5-1.5b_lora")
ap.add_argument("--draft_dir", default="adapters_export/draft_qwen2.5-0.5b_lora")
ap.add_argument("--private", action="store_true")
a = ap.parse_args()
api = HfApi(token=os.environ.get("HF_TOKEN"))
for d, name in ((a.target_dir, "trialmatch-qwen2.5-1.5b-lora"), (a.draft_dir, "trialmatch-qwen2.5-0.5b-draft-lora")):
    repo = f"{a.user}/{name}"
    api.create_repo(repo, exist_ok=True, private=a.private)
    api.upload_folder(folder_path=d, repo_id=repo, commit_message="Upload LoRA adapter (TrialGPT criterion screening)")
    print("uploaded:", f"https://huggingface.co/{repo}")
