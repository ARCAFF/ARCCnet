FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Update system and install tools
RUN apt-get update && apt-get upgrade -y && \
    apt-get install software-properties-common wget curl git htop neofetch \
    nano python3-dev python3-pip python3-wheel python3-setuptools -y && \
    # Update CUDA Linux GPG repository key
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    rm cuda-keyring_1.0-1_all.deb && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin && \
    mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub && \
    add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" -y && \
    apt-get update && \
    # Install recommended packages
    apt-get install zlib1g g++ freeglut3-dev \
    libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev libfreeimage-dev -y

# Copy the requirements.txt file into the container
COPY requirements.txt /tmp/

# Install Python libraries from requirements.txt
RUN python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Cleanup
RUN pip3 cache purge && \
    apt-get autoremove -y && \
    apt-get clean

# keep docker container alive
CMD ["tail", "-f", "/dev/null"]
