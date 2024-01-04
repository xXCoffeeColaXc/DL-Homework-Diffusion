import streamlit as st
from PIL import Image
import torch
from ddim_modules import UNet
import numpy as np


config = {
  "model_config": {
    "c_in": 3,
    "c_out": 3,
    "crop_size": 64,
    "image_size": 64,
    "conv_dim": 64,
    "block_depth": 3,
    "time_emb_dim": 256
  },
  "training_config": {
    "batch_size": 8,
    "epochs": 200,
    "lr": 0.0003,
    "beta_start": 0.0001,
    "beta_end": 0.02,
    "s_parameter": 0.008,
    "cos_scheduler": 0, 
    "noise_steps": 1000,
    "resume_epoch": False
  },
  "miscellaneous": {
    "num_workers": 1,
    "mode": "train",
    "wandb": 0, 
    "visualize": 0 
  },
  "directories": {
    "image_dir": "data/CUB_200_2011/CUB_200_2011",
    "metadata_dir": "data/metadata",
    "log_dir": "outputs/logs",
    "model_save_dir": "outputs/models",
    "sample_dir": "outputs/samples",
    "result_dir": "outputs/results"
  },
  "step_size": {
    "log_step": 10,
    "sample_step": 1000,
    "validation_step": 100,
    "model_save_step": 5,
    "ddim_sample_step": 10
  },
  "num_images" : 2,
}
device = "cuda" if torch.cuda.is_available() else "cpu"



def build_model(checkpoint_path):
    # Create
    unet = UNet(config["model_config"]["c_in"], 
                config["model_config"]["c_out"], 
                config["model_config"]["image_size"], 
                config["model_config"]["conv_dim"], 
                config["model_config"]["block_depth"], 
                config["model_config"]["time_emb_dim"])
    
    # Load
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    unet.load_state_dict(checkpoint['model_state_dict'])

    # # Compute alpha, beta, alpha_hat
    # beta = prepare_noise_schedule().to(device)
    # alpha = 1.0 - beta
    # alpha_hat = torch.cumprod(alpha, dim=0)

    unet.to(device)
    return unet

unet = build_model("outputs/models/198-checkpoint.ckpt")

def prepare_noise_schedule():
    return torch.linspace(config["training_config"]["beta_start"], 
                          config["training_config"]["beta_end"], 
                          config["training_config"]["noise_steps"])

beta = prepare_noise_schedule().to(device)
alpha = 1.0 - beta
alpha_hat = torch.cumprod(alpha, dim=0)


    
def forward_process(x, noise, t):
    """Performs the forward diffusion process.

    Args:
        x: Original image tensor.
        noise: Noise tensor to be added to the image.
        t: Time step for the diffusion process.

    Returns:
        Tensor of noised images.
    """
    sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:, None, None, None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - alpha_hat[t])[:, None, None, None]
    return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise

def ddpm_sample( n):
    print(f"Sampling {n} new images...") # TODO logger
    unet.eval()
    with torch.no_grad():
        x = torch.randn((n, 3, config["model_config"]["image_size"], config["model_config"]["image_size"])).to(device)
        for i in reversed(range(1, config["training_config"]["noise_steps"])):
            t = (torch.ones(n) * i).long().to(device) # create a tensor of lenght n with the current timestep
            # beta = prepare_noise_schedule().to(device)
            # alpha = 1.0 - beta
            # alpha_hat = torch.cumprod(alpha, dim=0)
            
            one_minus_alpha_hat = 1.0 - alpha_hat[t][:, None, None, None]
            if unet.requires_alpha_hat_timestep:
                predicted_noise = unet(x, one_minus_alpha_hat)
            else:
                predicted_noise = unet(x, t)
            alpha = alpha[t][:, None, None, None]
            alpha_hat = alpha_hat[t][:, None, None, None]
            beta = beta[t][:, None, None, None]
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
            
    
    denormed_x = [denorm(i.cpu().detach()) for i in x]
    return denormed_x


def ddim_sample(n, sample_steps):
    """Performs sampling of images using DDIM method.

    Args:
        n: Number of images to sample.

    Returns:
        Torch tensor of sampled images.
    """
    sample_step_number = sample_steps
    sample_steps = np.linspace(config["training_config"]["noise_steps"] - 1, 1, sample_step_number + 1, dtype=int)
    sample_steps = sample_steps[:-1] # drop the 1

    #print(f"Sampling {n} new images...")
    unet.eval()
    with torch.no_grad():
        x = torch.randn((n, 3, config["model_config"]["image_size"], config["model_config"]["image_size"])).to(device)
        for idx, i in enumerate(sample_steps):
            t = (torch.ones(n) * i).long().to(device) # create a tensor of lenght n with the current timestep
            alpha_hat_ = alpha_hat[t][:, None, None, None]
            one_minus_alpha_hat = 1.0 - alpha_hat_
            if unet.requires_alpha_hat_timestep:
                predicted_noise = unet(x, one_minus_alpha_hat)
            else:
                predicted_noise = unet(x, t)
            pred_img = (x - (torch.sqrt(one_minus_alpha_hat)) * predicted_noise) / torch.sqrt(alpha_hat_)
            if idx < len(sample_steps) - 1:
                next_t = (torch.ones(n) * sample_steps[idx+1]).long().to(device)
            x = forward_process(pred_img, predicted_noise, next_t)

    denormed_x = [denorm(i.cpu().detach()) for i in pred_img]
    return denormed_x

def denorm(image):
    mean = torch.tensor([0.4865, 0.4998, 0.4323])
    std = torch.tensor([0.2326, 0.2276, 0.2659])
    mean_expanded = mean.view(3, 1, 1).cpu().detach()
    std_expanded = std.view(3, 1, 1).cpu().detach()

    # Denormalize
    x_adj = (image * std_expanded + mean_expanded) * 255
    x_adj = x_adj.clamp(0, 255).type(torch.uint8)
    return x_adj


def generate_image():
    st.write("Generating image with steps:", config["training_config"]["noise_steps"])
    st.write("Generating image with sample steps:", config["step_size"]["ddim_sample_step"])
    st.write("Generating number of images:", config["num_images"])
    
    sampled_images = ddim_sample(n=config["num_images"], sample_steps=config["step_size"]["ddim_sample_step"])

    resized_images = []
    width, height = 32, 32  

    for image in sampled_images:
        pil_image = Image.fromarray(image.permute(1, 2, 0).byte().numpy())
        resized_image = pil_image.resize((width, height))
        resized_images.append(resized_image)

    # Display images in a row
    cols = st.columns(config["num_images"])  # create 2 columns for 2 images
    for i, col in enumerate(cols):
        col.image(resized_images[i], caption=f"Image {i+1}", use_column_width=True)


st.title("Diffusion Model Image Generator")

# Sliders
config["training_config"]["noise_steps"] = st.slider("Diffusion Step", 10, 1000, 10)
configconfig["step_size"]["ddim_sample_step"] = st.slider("Sample Step", 10, 200, 10)
config["num_images"] = st.slider("Number of images", 1, 6, 2)

# Button
if st.button("Generate"):
    generate_image()
