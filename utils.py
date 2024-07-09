import torch
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



def get_loaders(train_img_dir, train_mak_dir, val_img_dir, val_mask_dir,
                train_transform, val_transform,
                batch_size, num_workers, pin_memory=True):

    train_ds = CarvanaDataset(image_dir=train_img_dir, mask_dir=train_mak_dir, transform=train_transform)
    val_ds = CarvanaDataset(image_dir=val_img_dir, mask_dir=val_mask_dir,transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size = batch_size, num_workers = num_workers,pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size = batch_size, num_workers = num_workers,pin_memory = pin_memory)

    return train_loader, val_loader