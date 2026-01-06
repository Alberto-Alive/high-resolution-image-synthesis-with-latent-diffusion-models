import os
import torch
from safetensors.torch import save_file, load_file

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    
def save_ckpt(path: str, model: torch.nn.Module, optim: torch.optim.Optimizer | None = None, extra: dict | None = None):
    obj = {"model": model.state_dict()}
    if optim is not None:
        obj["optim"] = optim.state_dict()
    if extra:
        obj.update(extra)
    torch.save(obj, path)
    
def 