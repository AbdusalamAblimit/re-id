import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from IPython import embed 

def read_image(img_path):
    if not os.path.exists(img_path):
        raise IOError(f"{img_path} does not exist.")
    img = Image.open(img_path).convert('RGB')
    return img


class ImageDataset(Dataset):
    def __init__(self, dataset, transform = None):
        self.dataset = dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img,pid,camid


if __name__=='__main__':
    import data_manager
    data = data_manager.Market1501()
    train_data = data.train
    train_dataset = ImageDataset(train_data)
    embed()
    
