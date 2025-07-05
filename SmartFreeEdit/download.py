import os
from huggingface_hub import snapshot_download

# download hf models
SmartFreeEdit_path = "/data1/sqq/SmartFreeEdit"
if  not os.path.exists(SmartFreeEdit_path):
    SmartFreeEdit_path = snapshot_download(
        repo_id="lllyasviel/ControlNet",
        local_dir=SmartFreeEdit_path,
        token=os.getenv("HF_TOKEN"),
    )

print("Downloaded BrushEdit to ", SmartFreeEdit_path)
