#!/usr/bin/env python3
"""
Script to download required models for the inference server.
Run this after cloning the repository to set up all models.
"""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("❌ huggingface_hub not found. Install with: pip install huggingface_hub")
    sys.exit(1)

# Model configurations
MODELS = {
    "models/llm/tinyllama": {
        "repo": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "description": "Fast, small model for testing (1.1B parameters)"
    },
    "models/llm/phi-2": {
        "repo": "microsoft/phi-2",
        "description": "High-quality model for production (2.7B parameters)"
    },
    "models/minilm": {
        "repo": "sentence-transformers/all-MiniLM-L6-v2",
        "description": "Embedding model for vector memory (22M parameters)"
    }
}

def download_model(local_dir: str, repo_id: str, description: str):
    """Download a single model."""
    print(f"📥 Downloading {description}")
    print(f"   From: {repo_id}")
    print(f"   To: {local_dir}")

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=['*.bin', '*.msgpack', '*.h5', '*.ot', '*.md']
        )
        print(f"✅ Successfully downloaded to {local_dir}")
    except Exception as e:
        print(f"❌ Failed to download {repo_id}: {e}")
        return False
    return True

def main():
    print("🤖 Inference Server Model Downloader")
    print("=" * 40)

    # Create models directory
    os.makedirs("models", exist_ok=True)

    success_count = 0
    total_count = len(MODELS)

    for local_dir, config in MODELS.items():
        print(f"\n🔄 [{success_count + 1}/{total_count}] Processing {local_dir}")
        if download_model(local_dir, config["repo"], config["description"]):
            success_count += 1
        print()

    print("🎉 Download Summary:")
    print(f"   ✅ Successful: {success_count}/{total_count}")
    print(f"   ❌ Failed: {total_count - success_count}/{total_count}")

    if success_count == total_count:
        print("\n🚀 All models downloaded! You can now run:")
        print("   cargo run --bin cli -- --list-models")
    else:
        print("\n⚠️  Some models failed to download. You can try again or download manually.")

if __name__ == "__main__":
    main()
