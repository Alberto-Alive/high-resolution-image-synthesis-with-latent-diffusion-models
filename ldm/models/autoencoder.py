import torch
import torch.nn as nn
import torch.nn.functional as F


# a convolution layer as a tiny sliding window (like a 3x3 patch) that moves over
#  the image and computes a "match score". ex: an edge is a sudden change in brightness/color (dark ->light)
#  and a simple edge filter might look (conceptually) like "substract left from right":
#  if left side of the 3x3 patch is dark and the right side is bright, the filter outputs a big positive number => "edge here"
def conv(in_c, out_c, k=3, s=1, p=1):
    return nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p)



