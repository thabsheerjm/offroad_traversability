import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class GOOSEDataset(Dataset):
    def __init__(self, rgb_dir, mask_dir, mapping_csv,traversable_labels, transform=None):
        self.rgb_dir = rgb_dir 
        self.mask_dir = mask_dir
        self.img_ids = sorted(os.listdir(rgb_dir))
        self.transform = transform
        
        df = pd.read_csv(mapping_csv)
        self.traversable_ids = set(df[df["class_name"].isin(traversable_labels)]["label_key"].values)

    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        rgb_path = os.path.join(self.rgb_dir, img_id)
        mask_filename = img_id.replace("_windshield_vis", "_labelids")
        mask_path = os.path.join(self.mask_dir, mask_filename)

        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # print(np.unique(mask))
        resize_size = (520, 520)
        rgb = cv2.resize(rgb, resize_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, resize_size, interpolation=cv2.INTER_NEAREST)
        binary_mask = np.isin(mask, list(self.traversable_ids)).astype(np.uint8) 

        if self.transform:
            augmented = self.transform(image=rgb, mask=binary_mask)
            rgb = augmented['image']
            binary_mask = augmented['mask']

        rgb = transforms.ToTensor()(rgb)
        mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0).float()
        return rgb, mask_tensor 
        

if __name__ == '__main__':

    dataset = GOOSEDataset(
    rgb_dir="data/GOOSE/train/rgb",
    mask_dir="data/GOOSE/train/masks",
    mapping_csv="data/GOOSE/goose_label_mapping.csv",
    traversable_labels=["asphalt", "gravel", "low_grass", "high_grass", "soil",
    "cobble", "moss", "bikeway", "sidewalk", "road_marking"])

    img, mask = dataset[0]
    print(img.shape, mask.shape) 