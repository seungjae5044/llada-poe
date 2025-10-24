#!/usr/bin/env python3
"""모델을 미리 다운로드하는 스크립트"""

from huggingface_hub import snapshot_download

print("Downloading GSAI-ML/LLaDA-1.5...")
snapshot_download(
    repo_id="GSAI-ML/LLaDA-1.5",
    local_dir_use_symlinks=False,
)

print("\nDownloading GSAI-ML/LLaDA-8B-Instruct...")
snapshot_download(
    repo_id="GSAI-ML/LLaDA-8B-Instruct",
    local_dir_use_symlinks=False,
)

print("\nAll models downloaded successfully!")
