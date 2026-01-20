import argparse
import torch
from torchvision.utils import save_image

from models.autoencoder import AutoencoderKL
from models.unet import UNetEps
from diffusion.schedules import linear_bet_schedule, make_ddpm_constants
from diffusion.ddpm import DDPM
from utils import ensure_dir, load_ckpt

@torch.no_grad()
def nmain(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_dir(args.out)
    
    ae = AutoencoderKL(z_channels=args.z, base=args.ae_base).to(device)
    
    load_ckpt(args.ae_ckpt, ae, map_location=device)
    ae.eval()