import torch
import torch.nn as nn
from tqdm import tqdm
import config
from utils import *
from dataloader import get_data
from torchvision.utils import save_image
from modules import UNet
from torch import optim
import os

class Diffusion:
    def __init__(self):
        # Compute alpha, beta, alpha_hat
        self.beta = self.prepare_noise_schedule().to(config.DEVICE)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)


    # TODO Implement CosineScheduler
    def prepare_noise_schedule(self):
        return torch.linspace(config.BETA_START, config.BETA_END, config.NOISE_STEPS)
    
    # Add t step noise to an image / noise_images
    # x(t) = sqrt(alpha_hat)*x(0) + sqrt(1-alpha_hat)*epsilon
    def forward_process(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x) # random noise
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon
    
    # sample a random timestep
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=config.NOISE_STEPS, size=(n,))

    # NOTE Algorithm 2 Sampling from original paper
    def sample(self, model, n):
        print(f"Sampling {n} new images...") # TODO logger
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)).to(config.DEVICE)
            for i in tqdm(reversed(range(1, config.NOISE_STEPS)), position=0):
                t = (torch.ones(n) * i).long().to(config.DEVICE) # create a tensor of lenght n with the current timestep
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        # bringing the values back to valid pixel range
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    # NOTE Algorithm 1 Traning from original paper
    def train(self, model, opt, mse, dataloader):
        #setup_logging(args.run_name) TODO logger
        #logger = SummaryWriter(os.path.join("runs", args.run_name))
        l = len(dataloader)

        for epoch in range(1, config.NUM_EPOCHS):
            #logging.info(f"Starting epoch {epoch}:") TODO logger
            print(f"Starting epoch {epoch}:")
            pbar = tqdm(dataloader)
            for i, (images, label, _) in enumerate(pbar):
                images = images.to(config.DEVICE)
                t = self.sample_timesteps(images.shape[0]).to(config.DEVICE) # get batch amount random timesteps
                x_t, noise = self.forward_process(images, t) # add t timestep noise to the image
                predicted_noise = model(x_t, t) # backward process: predict the added noise
                loss = mse(noise, predicted_noise) # calculate loss

                opt.zero_grad()
                loss.backward()
                opt.step()

                pbar.set_postfix(MSE=loss.item())
                #logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i) TODO logger
            print(f"Epoch: {epoch} MSE: {loss.item()}")

            if epoch % config.SAMPLE_STEP == 0:
                sampled_images = self.sample(model, n=images.shape[0])
                # TODO save_image from torch
                # TODO save_model
                # TODO load_model
                save_images(sampled_images, os.path.join(config.EVAL, config.RUN_NAME, f"{epoch}.jpg"))
                #torch.save(model.state_dict(), os.path.join("models", config.RUN_NAME, f"ckpt.pt"))


if __name__ == '__main__':
    model = UNet().to(config.DEVICE)

    # TODO load model
    # ckpt = torch.load("./working/orig/ckpt.pt")
    # model.load_state_dict(ckpt)
    
    diffusion = Diffusion()
    dataloader = get_data(type='train')
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    mse = nn.MSELoss()

    diffusion.train(model=model, opt=optimizer, mse=mse, dataloader=dataloader)

    # x = diffusion.sample(model, 2)
    # print(x.shape)
    # plt.figure(figsize=(32, 32))
    # plt.imshow(torch.cat([
    #     torch.cat([i for i in x.cpu()], dim=-1),
    # ], dim=-2).permute(1, 2, 0).cpu())
    # plt.show()