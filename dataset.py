'''
This code is to load images and masks: data loader

Input:
-Images and Masks  (CH,H,W)

Output:
- Images after transformations and convert to float tensor (CH,H,W)
''' 

import torch
import numpy as np
from torch.utils.data import Dataset
from skimage.color import rgb2gray


class ImagesDataset(Dataset):
    def __init__(self, img_paths: list, channels:list, transform=None, mode='train', limit=None):
        self.img_paths = img_paths
        self.channels =channels
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
            

        img = load_image(img_file_name,self.channels )
        #print(self.mode)

        if self.mode == 'train':
            mask = load_mask(img_file_name)

            img, mask = self.transform(img, mask)
                        
            maskTensor = None
            
            if(mask.ndim == 2):
                maskTensor = torch.from_numpy(np.expand_dims(mask, 0)).float()
            else:
                maskTensor = torch.from_numpy(mask.transpose((2, 0, 1))).float()
            
            return to_float_tensor(img), maskTensor
        else:
            mask = np.zeros(img.shape[:2])
            img, mask = self.transform(img, mask)

            return to_float_tensor(img), str(img_file_name) 


def to_float_tensor(img):
    img=torch.from_numpy(np.moveaxis(img, -1, 0)).float()  
    return img


def load_image(path,channels): #in CH, H,W  out: H,W,CH
    img = np.load(str(path))
    img=img.transpose((1, 2, 0))  
    return  img 

def mask_to_onehot(mask):

    maskgrey = rgb2gray(mask)
    #Constant values for grey mask classes (terrain,structures,roads)
    unique = [0.15043333333333334,0.5098819607843137,0.8802066666666667]

    #Getting 1 channel index array
    count = 0
    for v in unique:
        maskgrey[maskgrey==v]=count
        count+=1
        
    #Converting index array to one hot
    maskgrey = maskgrey.astype(np.int64)
    maskgrey = torch.nn.functional.one_hot(torch.from_numpy(maskgrey),num_classes=3)
    maskgrey = maskgrey.numpy()
    
    mask = maskgrey # Multiclass
    #mask = maskgrey[:,:,1] #select structure class
    #mask = maskgrey[:,:,0] #select terrain class
    #mask = maskgrey[:,:,2] #select road class
    return mask

def load_mask(path):   #H,W,CH   
    mask = np.load(str(path).replace('images', 'masks').replace(r'.npy', r'_a.npy'), 0)
    #mask=mask.reshape(mask.shape[1],-1)
#    mask =np .max(mask, axis=2)  #convert of 3 channel to 1 channel
#    mask=(mask > 0).astype(np.uint8)
#    return mask

#### probar con 3 clases
    mask=mask.transpose(1, 2, 0)#.reshape(mask.shape[1],-1)
    mask = mask_to_onehot(mask) #mask = (mask[:,:,1]> 0).astype(np.uint8) 
    return mask
