import torch
import torch.nn.functional as F


class DDPM:
    def __init__(self, eps_model, consts: dict):
        self.eps_model = eps_model
        self.c = consts
        
    def q_sample(self, x0)