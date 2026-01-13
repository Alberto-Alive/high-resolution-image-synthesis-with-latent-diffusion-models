import torch
import torch.nn.functional as F


class DDPM:
    def __init__(self, eps_model, consts: dict):
        self.eps_model = eps_model
        self.c = consts
        
    def q_sample(self, x0, t, noise=None):
        '''Basically here we return noised up image and the gaussian noise added to that image
            so the purpose of the model is to learn gaussian noise relative to (meshed into/combined with) different labelled images.
            Order has a weird property when combined with disorder as it creates novelty'''
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.c["sqrt_alphas_cumprod"][t].view(-1,1,1,1)
        sqrt_om = self.c["sqrt_one_minus_alphas_cumprod"][t].view(-1,1,1,1)
        
        return sqrt_ab * x0 + sqrt_om * noise, noise
    
    def loss(self, x0, t):
        '''Here we just measure how far off was the model in distinguishing/predicting noise from actual data'''
        xt, eps =self.q_sample(x0, t)
        eps_pred =self.eps_model(xt, t)
        return F.mse_loss(eps_pred, eps)
    
    @torch.no_grad()
    def p_sample(self, xt, t_int):
        b = xt.shape[0]