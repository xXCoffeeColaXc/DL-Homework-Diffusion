# DL Homework: Diffusion

# Team Name
S+

## Team Members
- Tamásy András - Neptun Code: EA824Y
- Takáts Bálint - Neptun Code: PUWI1T
- Hujbert Patrik - Neptun Code: D83AE5

![Forward process](forwardProcess.png)
![Backward process](backwardDenoising.png)


## Project Description
This project focuses on Denoising Diffusion Probabilistic Models (DDPMs) to generate bird images. Leveraging the comprehensive CUB200 dataset, which contains diverse and detailed images of bird species, the project aims to harness the power of diffusion models for generative tasks.

## Repository Files & Functions
- `config.py`: This file contains all configuration settings and parameters required for the project.
- `create_metadata.py`:  Responsible for generating and saving metadata in the form of JSON files. 
- `dataloader.py`: Handles the loading of data for the diffusion model
- `ddpm.py`: This file implements the Denoising Diffusion Probabilistic Model (DDPM). It includes the core functionalities for training and sampling from the diffusion model. The file defines the DDPM class with methods for the forward and reverse diffusion processes, loss computation, and utilities for handling the diffusion steps. It serves as the backbone of the diffusion-based generative model, enabling the generation of new data samples through a trained diffusion process.
- `main.py`: The main entry point of the application. This script handles command-line arguments and orchestrates the initialization and execution of the Diffusion model.
- `modules.py`: This file houses the architectural modules for the U-Net model used in the diffusion process. It includes definitions for various neural network layers and blocks, such as encoders, decoders, attention mechanisms, and other components specific to the U-Net architecture.
- `utils.py`: This file contains utility functions used across the project.

## Related Works
### Papers:
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)

### GitHub Repositories:
- [clear-diffusion-keras](https://github.com/beresandras/clear-diffusion-keras)


### Blog Posts:
- [Denoising Diffusion Implicit Models](https://keras.io/examples/generative/ddim/)
- [CUB-200-2011 (Caltech-UCSD Birds-200-2011)](https://paperswithcode.com/dataset/cub-200-2011)

## How to Run

### Downloading data:
1. Download the dataset from: https://data.caltech.edu/records/65de6-vp158
2. Ceate the following folders from the root of the project:
    - ./data
    - ./data/metadata
    - ./data/CUB_200_2011
3. Extract the bounding boxes and images into ./data/CUB_200_2011
4. Run create_metadata.py
5. You can also run the following commands from the project directory:
```bash
wget -O CUB_200_2011.tgz "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
tar -xzvf CUB_200_2011.tgz

wget -O segmentations.tgz "https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz?download=1"
tar -xzvf segmentations.tgz

mkdir data
cd data
mkdir CUB_200_2011
mkdir metadata
cd ..

mv CUB_200_2011 data/CUB_200_2011
mv segmentations data/CUB_200_2011

python create_metadata.py

```

### Building and Running the Container:
1. Clone the repository: `git clone [repository_link]`
2. Navigate to the repository directory: `cd [repo_name]`
3. Download the dataset as mentioned above
4. Build the Docker container: `docker build -t [container_name] .`
5. Run the Docker container with mounting the data: `docker run -v absolute/path/to/data:/home/custom_user/dl_homework_diffusion/data -it -p 8888:8888 [container_name]`

### Running the Pipeline Within the Container:
1. Once inside the container, navigate to the project directory: `cd /path/to/project/directory`
2. To train the model: `python main.py --mode=train`
3. To test the model: `python main.py --mode=test`

### Running data exploration Within the Container:
1. Once inside the container, navigate to the project directory: `cd /path/to/project/directory`
2. Start jupyter-lab: `jupyter-lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root`
3. Open jupyter-lab on host: `http://localhost:8888/`
4. Login with token
5. Open data exploration file: `data_exploration.ipynb`



### Training logs for Milestone 2:
1. https://api.wandb.ai/links/hakateam/s2jlnpab
