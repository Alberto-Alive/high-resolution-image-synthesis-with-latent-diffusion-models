import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import ImageFolderDataset
from models.autoencoder import AutoencoderKL, kl_loss, recon_loss
from utils import ensure_dir, save_ckpt

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load and normalise iamge size
    ds = ImageFolderDataset(args.data, size=args.size)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=True)
    
    ae = AutoencoderKL(z_channels=args.z)
