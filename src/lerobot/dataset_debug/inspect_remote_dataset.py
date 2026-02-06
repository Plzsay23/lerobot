import json
from huggingface_hub import hf_hub_download
import os
import sys

def get_repo_id_from_env():
    hf_user = os.environ.get("HF_USER")
    task_name = os.environ.get("TASK_NAME")
    
    if not hf_user or not task_name:
        print("Error: Environment variables 'HF_USER'or 'TASK_NAME'is not set.")
        sys.exit(1)
        
    return f"{hf_user}/{task_name}"

def main():
    REPO_ID = get_repo_id_from_env()
    print(f"Connecting to Hugging Face Hub: {REPO_ID}...")

    try:
        info_path = hf_hub_download(repo_id=REPO_ID, filename="meta/info.json", repo_type="dataset")
        stats_path = hf_hub_download(repo_id=REPO_ID, filename="meta/stats.json", repo_type="dataset")

        print(f"\n{'='*20} [1] info.json (Features) {'='*20}")
        with open(info_path, 'r') as f:
            info_data = json.load(f)
            if "features" in info_data:
                for key in info_data["features"].keys():
                    print(f"  - {key}")
            else:
                print("  [Warning] 'features' key not found in info.json")

        print(f"\n{'='*20} [2] stats.json (Real Keys) {'='*20}")
        with open(stats_path, 'r') as f:
            stats_data = json.load(f)
            camera_keys = []
            for key in stats_data.keys():
                print(f"  - {key}")
                if "image" in key:
                    camera_keys.append(key)

        print(f"\n{'='*20} [3] Diagnosis {'='*20}")
        print(f"Found Camera Keys: {camera_keys}")
        
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()