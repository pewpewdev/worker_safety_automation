# AI Demo Server
This application takes videos and keeps them running infinitely on videos from S3.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Important Notes](#important-notes)
- [Installation](#installation)
    - [Using Repo](#using-repo)
    - [Docker](#docker)


    
## Prerequisites
- You need to get a `.env` file from your supervisor which contains all the environment variables.
- This `.env` file contains AWS credentials for downloading videos from S3.
- For cloning the repo, it's better to add your SSH key.
- Install Docker, `nvidia-container-toolkit` or `nvidia-docker2` if using Docker Setup. 




## Installation

### Using Repo
1. Clone the repo:
    ```sh
    git clone git@source.syook.com:mlai/ai_for_safety.git
    # or 
    https://source.syook.com/mlai/ai_for_safety.git
    ```
2. Change to the directory:
    ```sh   
    cd ai_for_safety
    ```
3. Add your `.env` file inside `ai_for_safety`.
4. Create a new conda environment:
    ```sh
    conda create -n ai_demo_server python=3.10 -y
    ```
5. Activate the conda environment:
    ```sh
    conda activate ai_demo_server
    ```
6. Install CUDA and cuDNN as per official PyTorch instructions (https://pytorch.org/):
    ```sh
    conda install -c conda-forge cudatoolkit=11.8 cudnn=8.9
    ```
7. Install PyTorch as per instructions from (https://pytorch.org/):
    ```sh
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```
8. Check if PyTorch is using CUDA:
    ```python
    >>> import torch
    >>> torch.cuda.is_available()
    True
    >>> torch.cuda.device_count()
    1
    >>> torch.cuda.current_device()
    0
    >>> torch.cuda.device(0)
    <torch.cuda.device at 0x7efce0b03be0>
    >>> torch.cuda.get_device_name(0)
    'GeForce GTX 950M'
    ```
9. Install other dependencies from `requirements.txt`:
    ```sh
    pip install -r requirements.txt
    ```
10. Test your flow:
    ```sh
    python app.py
    ```

### Using Docker
The Docker file is present inside `ai_for_safety/AI_demo_server/docker/Dockerfile`.<br>
Please refer to the Dockerfile for more understanding.

1. Create an image from the Dockerfile:
    ```sh 
    docker build -t demo_ai_server_image -f Dockerfile .
    ```
2. Run the container:
    ```sh
    docker run --env-file .env --gpus all --name demo_ai_server_container -p 80:80 demo_ai_server_image
    ```

# Important Notes:
- If you are using a CPU instead of a GPU, you need to modify the `config/config.json` file to reflect this change. Ensure you set the appropriate `device` configuration.