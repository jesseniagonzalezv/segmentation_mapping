import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from pathlib import Path

data_path = Path('data')


class WaterDataset(Dataset):
    def __init__(self, img_paths: list, to_augment=False, transform=None, mode='train', limit=None):
        self.img_paths = img_paths
        self.to_augment = to_augment
        self.transform = transform
        self.mode = mode
        self.limit = limit

    def __len__(self):
        if self.limit is None:
            return len(self.img_paths)
        else:
            return self.limit

    def __getitem__(self, idx):
        if self.limit is None:
            img_file_name = self.img_paths[idx]
        else:
            img_file_name = np.random.choice(self.img_paths)

        img = load_image(img_file_name)

        if self.mode == 'train':
            mask = load_mask(img_file_name)

            img, mask = self.transform(img, mask)

            return to_float_tensor(img), torch.from_numpy(np.expand_dims(mask, 0)).float()
        else:
            mask = np.zeros(img.shape[:2])
            img, mask = self.transform(img, mask)

            return to_float_tensor(img), str(img_file_name)


def to_float_tensor(img):
    img=torch.from_numpy(img.transpose(2, 1, 0)).float()
    return img


def load_image(path):
    img = np.load(str(path))
    img=img.transpose((2, 1, 0)) 
    #print('one image in', img.shape)
    return  img 

def load_mask(path):   
    mask= np.zeros((512, 512))
    mask = np.load(str(path).replace('images', 'masks').replace(r'.npy', r'_a.npy'), 0)
    #print('one mask in ant T', mask.shape)
    mask=mask.transpose(2, 1, 0).reshape(512,-1)
    mask=(mask > 0).astype(np.uint8)
    #print('before the mask',mask.shape)
    return mask