# create output folders
# save and load checkpoints - model weights
# maintain an EMA copy (a "smoothed" version of weights) that often samples better
import os
import torch
from safetensors.torch import save_file, load_file #this is a safe/fast format for saving tensors

def ensure_dir(path: str):
    '''Ensure the directory exists: ex. ensure_dir("checkpoints")'''
    os.makedirs(path, exist_ok=True)
    
def save_ckpt(path: str, model: torch.nn.Module, optim: torch.optim.Optimizer | None = None, extra: dict | None = None):
    obj = {"model": model.state_dict()}
    if optim is not None:
        obj["optim"] = optim.state_dict()
    if extra:
        obj.update(extra)
    torch.save(obj, path)
    
def load_ckpt(path: str, model: torch.nn.Module, optim: torch.optim.Optimizer | None = None, map_location="cpu"):
    obj = torch.load(path, map_location=map_location)
    model.load_state_dict(obj["model"], strict=True)
    if optim is not None and "optim" in obj:
        optim.load_state_dict(obj["optim"])
    return obj
    
@torch.no_grad()
def ema_update(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float):
    msd = model.state_dict()
    for k, v in ema_model.state_dict().items():
        if k in msd:
            v.copy_(v * decay + msd[k] * (1.0 - decay))
            
