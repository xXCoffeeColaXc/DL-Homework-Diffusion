import streamlit as st
from PIL import Image
import torch
from ddim_modules import UNet
import numpy as np
import onnxruntime

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
onnx_unet = onnxruntime.InferenceSession("outputs/models/198-checkpoint.onnx")

def prepare_noise_schedule():
    return torch.linspace(config["training_config"]["beta_start"], 
                          config["training_config"]["beta_end"], 
                          config["training_config"]["noise_steps"])

def ddpm_sample( n):
    print(f"Sampling {n} new images...") # TODO logger
    unet.eval()
    with torch.no_grad():
        x = torch.randn((n, 3, config["model_config"]["image_size"], config["model_config"]["image_size"])).to(device)
        for i in reversed(range(1, config["training_config"]["noise_steps"])):
            t = (torch.ones(n) * i).long().to(device) # create a tensor of lenght n with the current timestep
            beta = prepare_noise_schedule().to(device)
            alpha = 1.0 - beta
            alpha_hat = torch.cumprod(alpha, dim=0)
            
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

def ddpm_sample_onnx(n):
    print(f"Sampling {n} new images...") # TODO logger
    unet.eval()
    with torch.no_grad():
        x = torch.randn((n, 3, config["model_config"]["image_size"], config["model_config"]["image_size"])).to(device)
        for i in reversed(range(1, config["training_config"]["noise_steps"])):
            t = (torch.ones(n) * i).long().to(device) # create a tensor of lenght n with the current timestep
            beta = prepare_noise_schedule().to(device)
            alpha = 1.0 - beta
            alpha_hat = torch.cumprod(alpha, dim=0)
            one_minus_alpha_hat = 1.0 - alpha_hat[t][:, None, None, None]
            if unet.requires_alpha_hat_timestep:
                predicted_noise = onnx_unet.run([], {'images':x.detach().cpu().numpy(), 'timestep': one_minus_alpha_hat.detach().cpu().numpy()})[0]
            else:
                predicted_noise = onnx_unet.run([], {'images':x.detach().cpu().numpy(), 'timestep': one_minus_alpha_hat.detach().cpu().numpy()})[0]
            predicted_noise = torch.from_numpy(predicted_noise)
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
    st.write("Generating number of images:", config["num_images"])
    
    if config["num_images"] == 2:
        sampled_images = ddpm_sample_onnx(n=config["num_images"])
    else:
        sampled_images = ddpm_sample(n=config["num_images"])

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
config["num_images"] = st.slider("Number of images", 2, 6, 2)

# Button
if st.button("Generate"):
    generate_image()
