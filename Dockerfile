# Start from the UBI8 base image with CUDA support
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Update the package list and install Python 3 and other dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    vim \
    git \
    python3 \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .


#Install pytorch
RUN pip3 install torch torchvision torchaudio

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


# Copy the files
COPY app.py /app
COPY config /app/config
COPY models /app/models
COPY utils /app/utils
COPY weights /app/weights
#COPY videos /app/videos

# Make port 80 available to the world outside this container
EXPOSE 80

# Disable Python output buffering
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["python3", "app.py"]
