import os
from typing import Optional

from dotenv import load_dotenv
from huggingface_hub import HfApi, login

load_dotenv()


def upload_model_to_huggingface(
    local_model_path: str,
    repo_name: str,
    token: Optional[str] = None,
    commit_message: Optional[str] = None,
    private: bool = True,
) -> Optional[str]:
    """
    Upload a trained model to Hugging Face Hub with smart handling of different model types.

    Args:
        local_model_path: Path to the saved model directory
        repo_name: Repository ID on HuggingFace (format: 'username/repo-name')
        token: HuggingFace API token, will use HF_TOKEN env variable if not provided
        commit_message: Custom commit message for the upload
        private: Whether to create a private repository

    Returns:
        str or None: URL of the uploaded model on HuggingFace Hub, or None if upload failed
    """
    # Get token from environment if not provided
    if token is None:
        token = os.getenv("HF_TOKEN")
        if token is None:
            raise ValueError("HF_TOKEN not provided as argument or environment variable")

    # Use default commit message if not provided
    if commit_message is None:
        commit_message = f"Upload fine-tuned model to {repo_name}"

    # Login to Hugging Face once at the beginning
    login(token=token)

    # Initialize the API
    api = HfApi(token=token)

    # Create or verify repository
    hub_url = f"https://huggingface.co/{repo_name}"
    try:
        api.repo_info(repo_id=repo_name)
        print(f"Repository {repo_name} already exists")
    except Exception:
        print(f"Creating new repository: {repo_name} (private={private})")
        api.create_repo(repo_id=repo_name, private=private, exist_ok=True)

    # Check if the path exists
    if not os.path.exists(local_model_path):
        raise ValueError(f"Model path does not exist: {local_model_path}")

    # Try different upload strategies in order of preference
    upload_success = False

    # Strategy 3: Direct folder upload as last resort
    if not upload_success:
        try:
            print("Falling back to direct folder upload...")
            api.upload_folder(
                folder_path=local_model_path, repo_id=repo_name, commit_message=commit_message
            )
            print(f"Successfully uploaded model folder to {hub_url}")
            upload_success = True
        except Exception as e:
            print(f"Folder upload failed: {e}")

    # Return result
    if upload_success:
        return hub_url
    else:
        print("All upload strategies failed.")
        return None


# Example usage
if __name__ == "__main__":
    LOCAL_MODEL_PATH = "/home/ronaksagtani/artifacts/spec-forge/outputs/eagle3-40k-acc-0p8-20250924_212817/epoch_0"
    REPO_NAME = "sourcegraph/eagle3-speculator-50k-amp-tab-draft-model"

    upload_model_to_huggingface(
        local_model_path=LOCAL_MODEL_PATH,
        repo_name=REPO_NAME,
    )
