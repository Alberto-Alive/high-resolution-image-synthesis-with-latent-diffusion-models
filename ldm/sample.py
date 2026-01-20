import argparse
import torch
from torchvision.utils import save_image

from models.autoencoder import AutoencoderKL
from models.unet import UNetEps
from diffusion.schedules import linear_beta_schedule, make_ddpm_constants
from diffusion.ddpm import DDPM
from utils import ensure_dir, load_ckpt

@torch.no_grad()
def nmain(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_dir(args.out)
    
    ae = AutoencoderKL(z_channels=args.z, base=args.ae_base).to(device)
    
    load_ckpt(args.ae_ckpt, ae, map_location=device)
    ae.eval()
    
    unet = UNetEps(in_ch=args.z, base=args.unet_base, tdim=args.tdim).to(device)
    load_ckpt(args.ldm_ckpt, unet, map_location=device)
    unet.eval()

    betas = linear_beta_schedule(args.T, device=device)
    consts = make_ddpm_constants(betas)
    ddpm = DDPM(unet, consts)
    
    zH = args.size // 8
    zW = args.size // 8
    z = ddpm.sample((args.n, args.z, zH, zW), T=args.T)
    z = z / args.latent_scale
    x = ae.decode(z)