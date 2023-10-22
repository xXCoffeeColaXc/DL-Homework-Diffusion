# DL Homework: Diffusion

# Team Name
S+

## Team Members
- Tamásy András - Neptun Code: EA824Y
- Takáts Bálint - Neptun Code: PUWI1T
- Hujbert Patrik - Neptun Code: D83AE5

## Project Description
[Provide a brief description of your project here. This should give readers an idea of what the diffusion model implementation accomplishes, any unique features, and its significance in the context of your university project.]

## Repository Files & Functions
- `config.py`: This file contains all configuration settings and parameters required for the project.
- `create_metadata.py`:  Responsible for generating and saving metadata in the form of JSON files. 
- `dataloader.py`: Handles the loading of data for the diffusion model

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

### Building and Running the Container:
1. Clone the repository: `git clone [repository_link]`
2. Navigate to the repository directory: `cd [repo_name]`
3. Download the dataset as mentioned above
4. Build the Docker container: `docker build -t [container_name] .`
5. Run the Docker container with mounting the data: `docker run -v absolute/path/to/data:/home/custom_user/dl_homework_diffusion/data -it -p 8888:8888 [container_name]`

### Running the dataloader Within the Container:
1. Once inside the container, navigate to the project directory: `cd /path/to/project/directory`
2. Run the dataloader file: `python dataloader.py`

### Running data exploration Within the Container:
1. Once inside the container, navigate to the project directory: `cd /path/to/project/directory`
2. Start jupyter-lab: `jupyter-lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root`
3. Open jupyter-lab on host: `http://localhost:8888/`
4. Login with token
5. Open data exploration file: `data_exploration.ipynb`


