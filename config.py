import torch
from torchvision import transforms

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
ROOT_DIR = "./data/CUB_200_2011/CUB_200_2011"
METADATA_DIR = "./data/metadata"


IMAGE_SIZE = 64

'''
NUM_CLASSES = 200
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 2

CHANNEL_IMG = 3
NUM_EPOCHS = 200
MODEL_CHECKPOINT = "diff.pth.tar"
'''


transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        transforms.RandomHorizontalFlip(p=0.5), # messes up the bounding boxes
        #transforms.RandomRotation(10),
        #transforms.ColorJitter(brightness=0.2),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        #transforms.RandomCrop(size=(64, 64), padding=4),
        #transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),

        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])


