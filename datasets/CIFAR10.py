import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from ignition.data import data_loader_sample_count
import torchvision.datasets


class Unnormalize:
    """Converts an image tensor that was previously Normalize'd
    back to an image with pixels in the range [0, 1]."""
    def __init__(self, mean, std):
        self.mean = torch.Tensor(mean).view(3, 1, 1)
        self.std = torch.Tensor(std).view(3, 1, 1)

    def __call__(self, tensor):
        return torch.clamp(tensor*self.std + self.mean, 0., 1.)


class CIFAR10:
    classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    num_classes = len(classes)

    # CIFAR-10 data is often preprocessed using ZCA whitening but we
    # just normalize it using the mean and stddev over the training set.
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    normalize_transforms = [
        ToTensor(),
        Normalize(mean, std),
    ]

    unnormalize_transform = Unnormalize(mean, std)
    
    def __init__(self, batch_size=128, num_workers=4, augment=[]):
        """Loads the CIFAR10 train and test sets.
        
        Parameters
        ----------
        batch_size: int
        num_workers: int
        augment: list of torchvision.transforms objects
        """
        train_transform = Compose(augment + self.normalize_transforms)

        self.train_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=train_transform)
        
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, pin_memory=True)

        test_transform = Compose(self.normalize_transforms)
        
        self.test_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=test_transform)
        
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, pin_memory=True)
        
        print("Train examples:", data_loader_sample_count(self.train_loader))
        print(" Test examples:", data_loader_sample_count(self.test_loader))
