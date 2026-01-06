import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")

class ImageFolderDataset(Dataset):
    def __init__(self, root: str, size: int =256):
        self.files = []
        for dp, _, fn in os.walk(root):
            for f in fn:
                if f.lower().endswith(IMG_EXTS):
                    self.files.append(os.path.join(dp,f))
        if not self.files:
            raise ValueError(f"No images found under {root}")
        self.tfm = transforms.Compose([
            transforms.resize(size, interpolation=transforms.interpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3) # [0.5]*3 means it uses 0.5 for each of the 3 channels: R, G, B.
        ])
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx: int):
        p = self.files[idx]
        img = Image.open(p).convert("RGB")
        return self.tfm(img)