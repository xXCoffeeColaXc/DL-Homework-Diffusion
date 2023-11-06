import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader import CUB200Dataset
import config

# TODO denormalize
# 

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(type="train"):
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),

        transforms.RandomHorizontalFlip(p=0.5), # messes up the bounding boxes
        #transforms.RandomRotation(10),
        #transforms.ColorJitter(brightness=0.2),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        #transforms.RandomCrop(size=(64, 64), padding=4),
        #transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),

        transforms.ToTensor(), # Scaled data into [0, 1]
        transforms.Normalize(0.5, 0.5),
        #transforms.Lambda(lambda t: (t*2) - 1) # Scales between [-1, 1] TODO: denormalize (t+1)/2
    ])
    dataset = CUB200Dataset(config.ROOT_DIR, transform, split=type)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)
    return dataloader