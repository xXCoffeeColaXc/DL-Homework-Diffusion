import torch
from torchvision import transforms
import os

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
ROOT_DIR = "./data/CUB_200_2011/CUB_200_2011"
METADATA_DIR = "./data/metadata"
EVAL = './eval'
MODEL_CHECKPOINT = "diff.pth.tar"
RUN_NAME = "DDPM_BASELINE"

IMAGE_SIZE = 64
NOISE_STEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02

NUM_EPOCHS = 5
LEARNING_RATE = 3e-4 # TODO constant lr ?
BATCH_SIZE = 4
NUM_WORKERS = 2


SAMPLE_STEP = 1 # 


# transform = transforms.Compose([
#         transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

#         transforms.RandomHorizontalFlip(p=0.5), # messes up the bounding boxes
#         #transforms.RandomRotation(10),
#         #transforms.ColorJitter(brightness=0.2),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#         #transforms.RandomCrop(size=(64, 64), padding=4),
#         #transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),

#         transforms.ToTensor(), # Scaled data into [0, 1]
#         transforms.Normalize(0.5, 0.5),
#         transforms.Lambda(lambda t: (t*2) - 1) # Scales between [-1, 1] TODO: denormalize (t+1)/2
#     ])


