import os
import torch
from huggingface_hub import snapshot_download

from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, UniPCMultistepScheduler



torch_dtype = torch.float16
device = "cpu"

SmartFreeEdit_path = "/models"
brushnet_path = os.path.join(SmartFreeEdit_path, "checkpoint-100000/brushnet")
brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch_dtype)


base_models_list = [
    {
        "name": "henmixReal (Preload)",
        "local_path": "/models/base_model/henmixReal_v5c",
        "pipe": StableDiffusionBrushNetPipeline.from_pretrained(
            "/models/base_model/henmixReal_v5c", brushnet=brushnet, torch_dtype=torch_dtype, low_cpu_mem_usage=False
        ).to(device)
    },
    {
        "name": "meinamix (Preload)",
        "local_path": "/models/base_model/meinamix_meinaV11",
        "pipe": StableDiffusionBrushNetPipeline.from_pretrained(
            "/models/base_model/meinamix_meinaV11", brushnet=brushnet, torch_dtype=torch_dtype, low_cpu_mem_usage=False
        ).to(device)
    },
    {
        "name": "realisticVision (Default)",
        "local_path": "/models/base_model/realisticVisionV60B1_v51VAE",
        "pipe": StableDiffusionBrushNetPipeline.from_pretrained(
            "/models/base_model/realisticVisionV60B1_v51VAE", brushnet=brushnet, torch_dtype=torch_dtype, low_cpu_mem_usage=False
        ).to(device)
    },
]

base_models_template = {k["name"]: (k["local_path"], k["pipe"]) for k in base_models_list}
