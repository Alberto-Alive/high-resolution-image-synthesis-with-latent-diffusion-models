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
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    
    betas =linear_beta_schedule(args.T, device=device)
    consts = make_ddpm_constants(betas)
    ddpm = DDPM(unet, consts)
    
    ensure_dir(args.out)
    
    step = 0
    unet.train()
    for epoch in range(args.epochs):
        pbar = tqdm(dl, desc=f"LDM epoch {epoch}")
        for x in pbar:
            x = x.to(device, non_blocking=True)
            with torch.no_grad():
                z0 = encode_latents(ae, x, scale=args.latent_scale)
                
            t = torch.randint(0, args.T, (z0.shape[0],) device=device, dtype=torch.long)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                loss = ddpm.loss(z0, t)
            
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            
            ema_update(unet_ema, unet, decay=args.ema)
            
            
            if step % 50 == 0:
                pbar.set_postfix(loss=float(loss))
                
            if step % args.save_every == 0 and step > 0:
                save_ckpt(f"{args.out}/ldm_step{step}.pt", unet, opt, extra={"step": step, "epoch": epoch})
                save_ckpt(f"{args.out}/ldm_ema_step{step}.pt", unet_ema, None, extra={"step": step, "epoch": epoch})

            step += 1
            
    save_ckpt(f"{args.out}/ldm_final.pt", unet, opt, extra={"step": step, "epoch": args.epochs})
    save_ckpt(f"{args.out}/ldm_ema_final.pt", unet_ema, None, extra={"step": step, "epoch": args.epochs})
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--ae_ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, default="checkpoints_ldm")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--T", type=int, default=1000)

    parser.add_argument("--z", type=int, default=4)
    parser.add_argument("--ae_base", type=int, default=128)

    parser.add_argument("--unet_base", type=int, default=256)
    parser.add_argument("--tdim", type=int, default=512)

    parser.add_argument("--latent_scale", type=float, default=1.0)  # you can tune later
    parser.add_argument("--ema", type=float, default=0.999)

    parser.add_argument("--save_every", type=int, default=2000)