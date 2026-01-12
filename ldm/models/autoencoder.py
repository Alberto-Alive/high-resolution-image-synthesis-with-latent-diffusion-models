import torch
import torch.nn as nn
import torch.nn.functional as F


# a convolution layer as a tiny sliding window (like a 3x3 patch) that moves over
#  the image and computes a "match score". ex: an edge is a sudden change in brightness/color (dark ->light)
#  and a simple edge filter might look (conceptually) like "substract left from right":
#  if left side of the 3x3 patch is dark and the right side is bright, the filter outputs a big positive number => "edge here"
def conv(in_c, out_c, k=3, s=1, p=1):
    return nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p)

# process the image/feature map without changing its size
class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(32, c),
            nn.SiLU(),
            conv(c, c),
            nn.GroupNorm(32, c),
            nn.SiLU(),
            conv(c, c),
        )
    def forward(self, x):
        return x + self.net(x)

# shrink the spatial size (downsample) and learn richer features
class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            conv(in_c, out_c, 4,2, 1),
            ResBlock(out_c),
            ResBlock(out_c)
        )
    def forward(self, x): return self.net(x)
    
# grow the spatial size (upsample) and refine   
class Up(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode="nearest"),
            conv(in_c, out_c),
            ResBlock(out_c),
            ResBlock(out_c),
        )
    def forward(self, x): return self.net(x)
    
    
class Encoder(nn.Module):
    def __init__(self, z_channels=4, base=128):
        super().__init__()
        self.inp = conv(3, base)
        self.d1 = Down(base, base)
        self.d2 = Down(base, base*2)
        self.d3 = Down(base*2, base*4)
        self.mid = nn.Sequential(ResBlock(base*4), ResBlock(base*4))
        self.out = nn.Sequential(
            nn.GroupNorm(32, base *4),
            nn.SiLU(),
            conv(base*4, z_channels*2, 3, 1,1)
        )
    def forward(self, x):
        x = self.inp(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.mid(x)
        return self.out(x)
    

class Decoder(nn.Module):
    def __init__(self, z_channels=4, base=128):
        super().__init__()
        self.inp = conv(z_channels, base*4)
        self.mid = nn.Sequential(ResBlock(base*4), ResBlock(base*4))
        self.u3 = Up(base*4, base*2)
        self.u2 = Up(base*2, base)
        self.u1 = Up(base, base)
        self.out = nn.Sequential(
            nn.GroupNorm(32, base),
            nn.SiLU(),
            conv(base, 3, 3, 1, 1),
            nn.Tanh()
        )
        
    def forward(self, z):
        z = self.inp(z)
        z = self.mid(z)
        z = self.u3(z)
        z = self.u2(z)
        z = self.u1(z)
        return self.out(z)
    
class AutoencoderKL(nn.Module):
    def __init__(self, z_channels=4, base=128):
        super().__init__()
        self.enc = Encoder(z_channels=z_channels, base=base)
        self.dec = Decoder(z_channels=z_channels, base=base)
    
    def encode(self, x):
        h = self.enc(x)
        mean, logvar = torch.chunk(h, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z):
        return self.dec(z)
    
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        zrec = self.decode(z)
        return xrec, mean, logvar
    

