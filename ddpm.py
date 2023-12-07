import torch
import torch.nn as nn
from tqdm import tqdm
from utils import *
from torch import optim
import os
import wandb
import time
import datetime
# from metrics import KID
# from torchmetrics.image.fid import FrechetInceptionDistance
import numpy as np

#from modules import UNet
from ddim_modules import UNet


class Diffusion:
    def __init__(self, config, dataloader):

        self.config = config
        self.dataloader = dataloader

        self.build_model()

        # self.kid_metric = KID()
        # self.fid_metric = FrechetInceptionDistance(feature=2048, normalize=True)

        if self.config.wandb:
            self.setup_logger()

    def build_model(self):
        '''Create Unet'''
        # self.unet = UNet(self.config.c_in, self.config.c_out, 
        #                  self.config.conv_dim, self.config.block_depth, self.config.time_emb_dim)
        self.unet = UNet(self.config.c_in, self.config.c_out, self.config.image_size, self.config.conv_dim, self.config.block_depth, self.config.time_emb_dim)
        
        # Compute alpha, beta, alpha_hat
        self.beta = self.prepare_noise_schedule().to(self.config.device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.opt = optim.AdamW(self.unet.parameters(), lr=self.config.lr)
        self.mse = nn.MSELoss()
        
        #self.print_network()
        
        self.unet = self.unet.to(self.config.device)


    # TODO Implement CosineScheduler
    def prepare_noise_schedule(self):
        return torch.linspace(self.config.beta_start, self.config.beta_end, self.config.noise_steps)
    
    # Add t step noise to an image / noise_images
    # x(t) = sqrt(alpha_hat)*x(0) + sqrt(1-alpha_hat)*epsilon
    def forward_process(self, x, noise, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat[t])[:, None, None, None]
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise
    
    # sample a random timestep
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.config.noise_steps, size=(n,))

    def ddim_sample(self, n):

        sample_step_number = self.config.ddim_sample_step
        sample_steps = np.linspace(1, self.config.noise_steps - 1, sample_step_number + 1, dtype=int)
        sample_steps = sample_steps[1:] # drop the 1

        print(f"Sampling {n} new images...")
        self.unet.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.config.image_size, self.config.image_size)).to(self.config.device)
            for i in reversed(sample_steps):
                t = (torch.ones(n) * i).long().to(self.config.device) # create a tensor of lenght n with the current timestep
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                one_minus_alpha_hat = 1.0 - alpha_hat
                if self.unet.requires_alpha_hat_timestep:
                    predicted_noise = self.unet(x, one_minus_alpha_hat)
                else:
                    predicted_noise = self.unet(x, t)
                pred_img = (x - (torch.sqrt(one_minus_alpha_hat)) * predicted_noise) / torch.sqrt(alpha_hat)
                x = self.forward_process(pred_img, predicted_noise, t)

            self.unet.train()

        # mean = torch.tensor([0.5, 0.5, 0.5])
        # std = torch.tensor([0.5, 0.5, 0.5])
        mean = torch.tensor([0.4865, 0.4998, 0.4323])
        std = torch.tensor([0.2326, 0.2276, 0.2659])
        mean = mean.to(self.config.device)
        std = std.to(self.config.device)

        mean = mean[:, None, None]
        std = std[:, None, None]


        x = x * std + mean
        x = x * 255
        x = x.clamp(0, 255).type(torch.uint8)
        return x



    # NOTE Algorithm 2 Sampling from original paper
    def ddpm_sample(self, n):
        print(f"Sampling {n} new images...") # TODO logger
        self.unet.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.config.image_size, self.config.image_size)).to(self.config.device)
            for i in tqdm(reversed(range(1, self.config.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.config.device) # create a tensor of lenght n with the current timestep
                one_minus_alpha_hat = 1.0 - self.alpha_hat[t][:, None, None, None]
                if self.unet.requires_alpha_hat_timestep:
                    predicted_noise = self.unet(x, one_minus_alpha_hat)
                else:
                    predicted_noise = self.unet(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                break
        self.unet.train()
        # bringing the values back to valid pixel range
        # x = (x.clamp(-1, 1) + 1) / 2
        # x = (x * 255).type(torch.uint8)

        mean = torch.tensor([0.4865, 0.4998, 0.4323])
        std = torch.tensor([0.2326, 0.2276, 0.2659])
        mean = mean.to(self.config.device)
        std = std.to(self.config.device)

        mean = mean[:, None, None]
        std = std[:, None, None]


        x = x * std + mean
        x = x * 255
        x = x.clamp(0, 255).type(torch.uint8)
        return x

    # NOTE Algorithm 1 Traning from original paper
    def train(self):

        # Start training from scratch or resume training.
        start_epoch = 0
        if self.config.resume_epoch:
            start_epoch = self.restore_model(self.config.resume_epoch) 
            
        l = len(self.dataloader)

        for epoch in range(start_epoch, self.config.epochs):
            print(f"Starting epoch {epoch+1}:")
            start_time = time.time() # duration for one epoch
            
            for batch_idx, (images, label, _) in enumerate(self.dataloader):
                # print(batch_idx)
                images = images.to(self.config.device)
                t = self.sample_timesteps(images.shape[0]).to(self.config.device) # get batch amount random timesteps
                noise = torch.randn_like(images)
                x_t = self.forward_process(images, noise, t) # add t timestep noise to the image
                one_minus_alpha_hat = 1.0 - self.alpha_hat[t][:, None, None, None]
                if self.unet.requires_alpha_hat_timestep:
                    predicted_noise = self.unet(x_t, one_minus_alpha_hat)
                else:
                    predicted_noise = self.unet(x_t, t)
                mse_loss = self.mse(noise, predicted_noise) # calculate loss

                self.opt.zero_grad()
                mse_loss.backward()
                self.opt.step()

                loss = {}
                loss['MSE/loss'] = mse_loss

                num_iter = batch_idx + epoch * len(self.dataloader) + 1

                # Print out training information.
                if num_iter % self.config.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Iteration [{}/{}]".format(et, batch_idx+1, len(self.dataloader))
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                # Sample images and save them
                if num_iter % self.config.sample_step == 0:
                    sampled_images = self.ddim_sample(n=images.shape[0])
                    save_images(sampled_images, os.path.join(self.config.sample_dir, f"{num_iter}.jpg"))  # TODO save_image from torch

               
                # Log to wandb
                if self.config.wandb:
                    wandb.log({
                        "loss": mse_loss,
                        "epochs": epoch,
                        })
                    
            # Save model after every epoch
            # NOTE we dont want to save the model state mid batch, thats why we save it after an epoch
            if (epoch+1) % self.config.model_save_step == 0:
                self.save_model(epoch+1)

            if epoch == 50 or epoch == 80 or epoch == 100:
                self.save_model(epoch+1)
                sampled_images = self.ddim_sample(n=16)
                save_images(sampled_images, os.path.join(self.config.sample_dir, f"{num_iter}.jpg"))  # TODO save_image from torch

            start_epoch =+ 1


    def test(self):
        print("started_testing")
        # # Load the trained model.
        # if self.config.resume_epoch:
        #     self.restore_model(self.config.resume_epoch)

        # # num_iters = 0

        # for batch_idx, (images, label, _) in enumerate(self.dataloader):
        #     real_images = images.to(self.config.device) #[0, 1]
        #     generated_images = self.ddim_sample(self.config.batch_size) #[0, 255]
        #     generated_images = (generated_images / 255) #[0, 1]
            
        #     # TODO check what's wrong when batch_size=1, while updating the metrics
        #     self.kid_metric.update(real_images, generated_images)
        #     self.fid_metric.update(real_images, real=True)
        #     self.fid_metric.update(generated_images, real=False)

        #     # num_iters += 1
        #     # if num_iters > 2: 
        #     #     break
            
        # kid_score = self.kid_metric.compute()
        # print(f"KID score: {kid_score}")
        # fid_score = self.fid_metric.compute()
        # print(f"FID score: {fid_score}")
        # if self.config.wandb:
        #     print("wandb log")
        #     wandb.log({
        #         "kid_score": kid_score,
        #         "fid_score": fid_score
        #     })

        # self.kid_metric.reset()
        # self.fid_metric.reset()
        pass

    def save_model(self, epoch):
        save_dict = {
            'epoch': epoch,
            'model_state_dict': self.unet.state_dict(),
            'optimizer_state_dict': self.opt.state_dict()
        }

        save_path = os.path.join(self.config.model_save_dir, '{}-checkpoint.ckpt'.format(epoch))
        torch.save(save_dict, save_path)
        print('Saved checkpoints into {}...'.format(save_path))


    def restore_model(self, resume_epoch):
        print('Loading the trained model from epoch {}...'.format(resume_epoch))
        checkpoint_path = os.path.join(self.config.model_save_dir, '{}-checkpoint.ckpt'.format(resume_epoch))
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        self.unet.load_state_dict(checkpoint['model_state_dict'])
        self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        return start_epoch


    def print_network(self):
        """Print out the network information."""
        num_params = 0
        for p in self.unet.parameters():
            num_params += p.numel()
        print(self.unet)
        print("The number of parameters: {}".format(num_params))

    def setup_logger(self):
        # Initialize WandB
        wandb.init(project='bird-diffusion-project', config={
            "image_size": self.config.image_size,
            "block_depth": self.config.block_depth,
            "time_emb_dim": self.config.time_emb_dim,
            "batch_size": self.config.batch_size,
            "epochs": self.config.epochs,
            "lr": self.config.lr,
            "noise_steps": self.config.noise_steps,
            # ... Add other hyperparameters here
        })

        # Ensure DEVICE is tracked in WandB
        wandb.config.update({"device": self.config.device})
