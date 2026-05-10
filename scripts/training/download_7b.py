import os
from huggingface_hub import snapshot_download

def main():
    repo_id = "Qwen/Qwen2.5-7B-Instruct"
    # Go up 2 levels from scripts/training
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_dir = os.path.dirname(base_dir) 
    
    local_dir = os.path.join(base_dir, "models", "Qwen2.5-7B-Instruct")
    
    print(f"Downloading {repo_id} to {local_dir}")
    print("This will take a while, as the model is around 14-15GB.")
    
    # We only need the core model files, ignore heavy flax/gguf/safetensors if we just want torch
    # But for modern HF, safetensors are preferred. We'll ignore pt/bin if safetensors exist.
    ignore_patterns = ["*.pt", "*.bin"] # Assuming it uses safetensors
    
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        ignore_patterns=ignore_patterns,
        resume_download=True,
        max_workers=4
    )
    
    print("Download complete!")

if __name__ == "__main__":
    main()
