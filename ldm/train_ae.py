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
    # another place where we want to keep order close to disorder as this relation helps the 
    # model generalise and not overfit via the weight_decay mechanism that pulls weights towards 0
    # similar to how we add gaussian noise to images - we need this order-disorder relationship
    # in fact is nothing magic given the physical reality we live in, we need to find / create order 
    opt = torch.optim.AdamW(ae.parameters(), lr=args.lr, weight_decay=1e-4)