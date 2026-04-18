"""Pre-download k2-fsa/OmniVoice weights into HF_HOME so the server starts instantly."""
import os

os.environ.setdefault("HF_HOME", "/app/hf_cache")

from huggingface_hub import snapshot_download

token = os.environ.get("HF_TOKEN") or None

print("Downloading k2-fsa/OmniVoice …")
path = snapshot_download(
    repo_id="k2-fsa/OmniVoice",
    token=token,
)
print(f"Downloaded to: {path}")
