import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
ROOT_DIR = "data/CUB_200_2011/CUB_200_2011"
METADATA_DIR = "data/metadata"

'''
NUM_CLASSES = 200
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNEL_IMG = 3
NUM_EPOCHS = 200
MODEL_CHECKPOINT = "diff.pth.tar"
'''


'''
transform = transforms.Compose([
transforms.Resize((128, 128)),
transforms.ToTensor(),
])
'''


