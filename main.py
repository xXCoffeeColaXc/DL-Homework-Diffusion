import argparse
from ddpm import Diffusion
from config import Config
from dataloader import get_data
from torch.backends import cudnn
from utils import *
import wandb
from visualizer import DiffusionVisualizer
from app import VisualizerApplication

def main(config):
    # For fast training.
    cudnn.benchmark = True

    create_folders(config=config)

    dataloader = get_data(config.image_dir, config.metadata_dir, config.image_size, config.batch_size, config.mode, config.num_workers)
    config_obj = Config(config=config)
    
    diffusion = Diffusion(config=config_obj, dataloader=dataloader)

    if config.mode == 'train':
        print("Training...")
        diffusion.train()
    elif config.mode == 'test':
        if config.visualize:
            app = VisualizerApplication(config=config_obj)
            app.run()
            # print("Visualizing forward and backward process")
            
            # checkpoint_path = 'outputs/models/189-checkpoint.ckpt'
            # original_image = 'data/CUB_200_2011/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg'
            # visualizer = DiffusionVisualizer(cfg=config_obj, image_path=original_image, model_checkpoint_path=checkpoint_path)
            
            # # Visualize the noising process
            # noisy_images = visualizer.add_noise_for_steps(num_steps=10)
            # visualizer.visualize(noisy_images, "Forward Noising Process")

            # # Visualize the denoising process
            # denoised_images = visualizer.remove_noise_for_steps(num_steps=10)
            # visualizer.visualize(denoised_images, "Backward Denoising Process")



        else:
            print("Testing...")
            diffusion.test()
        

    if config.wandb:
        wandb.finish()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_in', type=int, default=3, help='dimension of input image')
    parser.add_argument('--c_out', type=int, default=3, help='dimension of output image')
    parser.add_argument('--crop_size', type=int, default=64, help='crop size')
    parser.add_argument('--image_size', type=int, default=64, help='image resolution')
    parser.add_argument('--conv_dim', type=int, default=64, help='number of conv filters in the first layer of the UNet')
    parser.add_argument('--block_depth', type=int, default=3, help='depth of conv layers in encoder/decoder')
    parser.add_argument('--time_emb_dim', type=int, default=256, help='number of channels for time embedding')
    
    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=8, help='mini-batch size')
    parser.add_argument('--epochs', type=int, default=200, help='number of total epochs')
    parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
    parser.add_argument('--beta_start', type=float, default=0.0001, help='beta start for noise scheluder')
    parser.add_argument('--beta_end', type=float, default=0.02, help='beta end for noise scheluder')
    parser.add_argument('--s_parameter', type=float, default=0.008, help='Sharpness parameter, controls how "sudden" the change is. A lower value makes the cosine curve smoother.')
    parser.add_argument('--cos_scheduler', type=int, choices=[0, 1], help='enable cosine scheduler, default is linear scheduler (0 for False, 1 for True)')
    parser.add_argument('--noise_steps', type=int, default=1000, help='noise steps for noise scheluder and sampling')
    parser.add_argument('--resume_epoch', type=int, default=None, help='resume training from this epoch')
   
    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test']) # TODO add val
    parser.add_argument('--wandb', type=int, choices=[0, 1], help='enable wandb logging (0 for False, 1 for True)')
    parser.add_argument('--visualize', type=int, choices=[0, 1], help='enable visualization (0 for False, 1 for True)')

    # Directories.
    parser.add_argument('--image_dir', type=str, default='data/CUB_200_2011/CUB_200_2011')
    parser.add_argument('--metadata_dir', type=str, default='data/metadata')
    parser.add_argument('--log_dir', type=str, default='outputs/logs')
    parser.add_argument('--model_save_dir', type=str, default='outputs/models')
    parser.add_argument('--sample_dir', type=str, default='outputs/samples')
    parser.add_argument('--result_dir', type=str, default='outputs/results')

    # Step size. NOTE This is in iterations, not in epochs!
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--validation_step', type=int, default=100) # NOTE validation function not implemented yet
    parser.add_argument('--model_save_step', type=int, default=5) # this one is in epochs
    parser.add_argument('--ddim_sample_step', type=int, default=5)

    config = parser.parse_args()
    print(config)
    main(config)