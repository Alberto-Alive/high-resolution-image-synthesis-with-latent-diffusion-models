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


class Down(nn.Module):
    def __init__(self, c, tdim):
        super().__init__()
        self.rb1 = ResBlock(c, tdim)
        self.rb2 = ResBlock(c, tdim)
        self.down = conv(c, c, 4, 2, 1)
        
    def forward(self, x, temb):
        x = self.rb1(x, temb)     
        x = self.rb2(x, temb)
        skip = x
        x = self.down(x)
        return x, skip

class Up(nn.Module):
    def __init__(self, c, tdim):
        super().__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),conv(c,c))
        self.rb1 = ResBlock(c, tdim)
        self.rb2 = ResBlock(c, tdim)
        
    def forward(self, x, skip, temb):
        x = self.up(x)
        x = x + skip
        x = self.rb1(x, temb)
        x = self.rb2(x, temb)
        return x
class UNetEps(nn.Module):
    def __init__(self, in_ch=4, base=256, tdim=512):
        super().__init__()
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmb(tdim),
            nn.Linear(tdim, tdim),
            nn.SiLU(),
            nn.Linear(tdim, tdim),
        )

        self.inp = conv(in_ch, base)
        self.d1 = Down(base, tdim)
        self.d2 = Down(base, tdim)
        self.mid1 = ResBlock(base, tdim)
        self.mid2 = ResBlock(base, tdim)
        self.u2 = Up(base, tdim)
        self.u1 = Up(base, tdim)

        self.out = nn.Sequential(
            nn.GroupNorm(32, base),
            nn.SiLU(),
            conv(base, in_ch),
        )

    def forward(self, x, t):
        temb = self.time_emb(t)

        x = self.inp(x)
        x, s1 = self.d1(x, temb)
        x, s2 = self.d2(x, temb)

        x = self.mid1(x, temb)
        x = self.mid2(x, temb)

        x = self.u2(x, s2, temb)
        x = self.u1(x, s1, temb)
        return self.out(x)