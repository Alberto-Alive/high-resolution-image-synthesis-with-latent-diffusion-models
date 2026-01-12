import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def conv(in_c, out_c, k=3, s=1, p=1):
    # small helper function for conv layers
    return nn.Conv2d(in_c, out_c, k, s, p)

class SinusoidalTimeEmb(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arrange(0, half, device=t.device) / half)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0,1))
        return emb
    
class ResBlock(nn.Module):
    def __init__(self, c, tdim):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, c)
        self.conv1 = conv(c, c)
        self.norm2 = nn.GroupNorm(32, c)
        self.conv2 = conv(c, c)
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(tdim, c))
        
    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm(x)))
        h = h+ self.time(temb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


        