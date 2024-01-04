import tkinter as tk
from PIL import Image, ImageTk
import torch
from tqdm import tqdm
from ddim_modules import UNet

class VisualizerApplication:
    def __init__(self, config) -> None:
        self.root = tk.Tk()
        self.config = config

        self.init_layout()

        checkpoint_path = 'outputs/models/198-checkpoint.ckpt'
        self.build_model(checkpoint_path)

    def build_model(self, checkpoint_path):
        # Create
        self.unet = UNet(self.config.c_in, self.config.c_out, self.config.image_size, self.config.conv_dim, self.config.block_depth, self.config.time_emb_dim)
        
        # Load
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        self.unet.load_state_dict(checkpoint['model_state_dict'])

        # Compute alpha, beta, alpha_hat
        self.beta = self.prepare_noise_schedule().to(self.config.device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.unet.to(self.config.device)

    def init_layout(self):
        # Create the main application window
        
        self.root.title("Diffusion Model Image Generator")
        self.root.geometry("800x600")  # Width x Height

        # Create frames for the layout
        self.left_frame = tk.Frame(self.root, width=400, height=600, bg='grey')
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.left_frame.pack_propagate(False)  # Prevent the frame from resizing to fit its contents

        self.right_frame = tk.Frame(self.root, width=400, height=600)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.right_frame.pack_propagate(False)  # Prevent the frame from resizing to fit its contents

        # Set up the grid layout in the left frame
        for i in range(10):  # Example: Creating 3 rows
            self.left_frame.grid_rowconfigure(i, weight=1)
            for j in range(10):  # Example: Creating 2 columns
                self.left_frame.grid_columnconfigure(j, weight=1)
                # Place a label as a placeholder in each grid cell
                #label = tk.Label(self.left_frame, text=f"{i},{j}", bg='yellow')
                #label.grid(row=i, column=j, sticky="nsew", padx=1, pady=1)


        self.diffusion_step_slider = tk.Scale(self.left_frame, from_=1, to=20, resolution=1, orient=tk.HORIZONTAL, label="Diffusion Step")
        self.diffusion_step_slider.grid(row=0, column=0, columnspan=10, sticky="ew", padx=10, pady=10)  # Span across the entire row


        self.beta_slider = tk.Scale(self.left_frame, from_=0.0001, to=0.02, resolution=0.0001, orient=tk.HORIZONTAL, label="Beta")
        self.beta_slider.grid(row=1, column=0, columnspan=10, sticky="ew", padx=10, pady=10)  # Span across the entire row

        # Add a Generate button in the last row, middle column
        self.generate_button = tk.Button(self.left_frame, text="Generate", command=self.generate_image)
        self.generate_button.grid(row=9, column=4, columnspan=2, pady=10, sticky="ew")  # Center the button


        # Set up the right frame for image display
        self.label_image_display = tk.Label(self.right_frame, text="Generated Image Will Appear Here", bg='black', fg='white')
        self.label_image_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def run(self):
        # Start the Tkinter event loop
        self.root.mainloop()

    # we can adjust that from the sliders
    def prepare_noise_schedule(self):
        return torch.linspace(self.config.beta_start, self.config.beta_end, self.config.noise_steps)

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
                
        
        denormed_x = [self.denorm(i.cpu().detach()) for i in x]
        return denormed_x
    
    def denorm(self, image):
        mean = torch.tensor([0.4865, 0.4998, 0.4323])
        std = torch.tensor([0.2326, 0.2276, 0.2659])
        mean_expanded = mean.view(3, 1, 1).cpu().detach()
        std_expanded = std.view(3, 1, 1).cpu().detach()

        # Denormalize
        x_adj = (image * std_expanded + mean_expanded) * 255
        x_adj = x_adj.clamp(0, 255).type(torch.uint8)
        return x_adj

    # Function to visualize images on the Tkinter right frame
    def visualize(self, images, title):
        # Clear previous images
        for widget in self.right_frame.winfo_children():
            widget.destroy()

        num_images = len(images)
        # Create a grid within the right frame
        rows = cols = int(num_images ** 0.5)
        for i in range(rows * cols):
            self.right_frame.grid_rowconfigure(i, weight=1)
            self.right_frame.grid_columnconfigure(i, weight=1)

        # Convert PyTorch images to PIL format and display them
        for i, image in enumerate(images):
            # Convert to PIL image
            pil_image = Image.fromarray(image.permute(1, 2, 0).byte().numpy())
            photo = ImageTk.PhotoImage(image=pil_image)
            label = tk.Label(self.right_frame, image=photo)
            label.image = photo  # Keep a reference, prevent garbage-collection
            # Calculate grid position
            row = i // cols
            col = i % cols
            label.grid(row=row, column=col, sticky="nsew", padx=2, pady=2)


    def generate_image(self):
        print("Generating image with beta:", self.beta_slider.get())
        print("Generating image with step:", self.diffusion_step_slider.get())

        sampled_images = self.ddpm_sample(n=6)
        self.visualize(sampled_images, "Generated Images")
















