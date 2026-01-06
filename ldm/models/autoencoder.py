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