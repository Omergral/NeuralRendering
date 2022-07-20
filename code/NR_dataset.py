import os
import cv2
import torch
from pathlib import Path
from torch.utils.data import Dataset


class MeshesDataset(Dataset):
    
    def __init__(self, images_dir, device: str = 'cuda'):
        super(MeshesDataset, self).__init__()
        self.images_dir = images_dir
        self.device = torch.device(device)

    def __len__(self):
        return len(os.listdir(self.images_dir))

    def __getitem__(self, item):
        item_dir = sorted(list(Path(self.images_dir).iterdir()), key=lambda x: int(str(x).split('/')[-1]))[item]
        rgb_img = torch.tensor(cv2.imread(str(item_dir / 'rgb.png'))).to(self.device).unsqueeze(0)
        sil_img = torch.tensor(cv2.imread(str(item_dir / 'sil.png'))).to(self.device).unsqueeze(0)
        blur_img = torch.tensor(cv2.imread(str(item_dir / 'blur.png'))).to(self.device).unsqueeze(0)

        return rgb_img, sil_img, blur_img
