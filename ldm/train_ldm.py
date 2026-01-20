import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import ImageFolderDataSet
from models.autoencoder import AutoencoderKL
from models.unet import UNetEps
from diffusion.schedules import linear_beta_schedule, make_ddpm_constants
from diffusion.ddpm import DDPM
from utils import ensure_dir, save_dir, save_ckpt, load_ckpt, ema_update


@torch.no_grad()
def encode_latents(ae, x, scale):
    mean, logvar = ae.encode(x)
    z = ae.reparameterize(mean, logvar)
    return z * scale


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = ImageFolderDataset(args.data, size=args.size)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0,pin_memory=True)
    
    ae.AutoencoderKL(z_channels=args.z, base=args.ae_base).to(device)
    load_ckpt(args.ae_ckpt, ae, map_location=device)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
        
    unet = UNetEps(in_ch=args.z, base=args.unet_base, tdim=args.tdim).to(device)
    unet_ema = UNetEps(in_ch=args.z, base=args.unet_base, tdim=args.tdim).to(device)
    unet_ema.load_state_dict(unet.state_dict())
    
    opt = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=1e-4)
    scalar = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    
    betas =linear_beta_s