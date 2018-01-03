from PIL import Image
import os, sys
import numpy as np
import torch.utils.data as data

class TinyImages(data.Dataset):
    """The 80 million tiny images dataset from http://horatio.cs.nyu.edu/mit/tiny/data/
    
    Only provides the images, not the labels.
    Can be a little slow since it directly seeks the file.
    
    Parameters
    ----------
    path: string
        Location of the tiny_images.bin file.
    count: int, optional)
        How many images to include.
    offset: int, optional
        The index of the first image that will be returned.
    transform: callable, optional
        A function that takes in an PIL image and returns a transformed version.
    """
    def __init__(self, path, count=79302017, offset=0, transform=None):
        self.f = open(path, "rb")
        self.count = count
        self.offset = offset
        self.transform = transform

    def __getitem__(self, index):
        self.f.seek((self.offset + index)*3*32*32, os.SEEK_SET)
        img = np.fromfile(self.f, dtype=np.uint8, count=3*32*32).reshape(3, 32, 32).transpose(2, 1, 0)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)        
        return img
    
    def __len__(self):
        return self.count
