import os
import torch
from openai import OpenAI

## init device
device = "cpu"
torch_dtype = torch.float16

vlms_list = [
    {
        "type": "openai",
        "name": "GPT4-o (Highly Recommended)",
        "local_path": "",
        "processor": "",
        "model": ""
    },
]

vlms_template = {k["name"]: (k["type"], k["local_path"], k["processor"], k["model"]) for k in vlms_list}