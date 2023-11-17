import torch
from torchvision import transforms
import os

# DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# ROOT_DIR = "./data/CUB_200_2011/CUB_200_2011"
# METADATA_DIR = "./data/metadata"
# EVAL = './eval'
# MODEL_CHECKPOINT = "diff.pth.tar"
# RUN_NAME = "DDPM_BASELINE"

# IMAGE_SIZE = 64
# NOISE_STEPS = 1000
# BETA_START = 1e-4
# BETA_END = 0.02

# NUM_EPOCHS = 100
# LEARNING_RATE = 3e-4 # TODO constant lr ?
# BATCH_SIZE = 1
# NUM_WORKERS = 1

# SAMPLE_STEP = 1 # 

class Config(object):
    def __init__(self, config) -> None:
        # Model configuration.
        self.c_in = config.c_in
        self.c_out = config.c_out
        self.crop_size = config.crop_size
        self.image_size = config.image_size
        self.conv_dim = config.conv_dim
        self.block_depth = config.block_depth
        self.time_emb_dim = config.time_emb_dim

        # Training configurations.
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.lr = config.lr
        self.beta_start = config.beta_start
        self.beta_end = config.beta_end
        self.noise_steps = config.noise_steps
        # self.resume_epoch = config.resume_epoch

        # Miscellaneous.
        self.num_workers = config.num_workers
        self.mode = config.mode
        self.wandb = config.wandb
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.image_dir = config.image_dir
        self.metadata_dir = config.metadata_dir
        self.log_dir = config.log_dir
        self.model_save_dir = config.model_save_dir
        self.sample_dir = config.sample_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.validation_step = config.validation_step
        self.model_save_step = config.model_save_step

