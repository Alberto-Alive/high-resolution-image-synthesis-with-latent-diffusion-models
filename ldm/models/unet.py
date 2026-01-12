import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def conv(in_c, out_c, k=3, s=1, p=1):
    # small helper function for conv layers
    return nn.Conv2d(in_c, out_c, k, s, p)

