# Use NVIDIA CUDA base image with development tools
FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/miniconda3/bin:${PATH}"

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    bzip2 \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    mkdir /root/.conda && \
    bash miniconda.sh -b && \
    rm miniconda.sh

# Set working directory
WORKDIR /app

# Copy the environment file first to leverage Docker caching
COPY environment.yml .
COPY submodules/ ./submodules/
# Create the conda environment
# We modify the environment.yml on the fly to ensure it matches the CUDA version
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

ENV TORCH_CUDA_ARCH_LIST="7.5"
RUN conda env create -f environment.yml && \
    conda clean -afy

# Ensure the shell uses the conda environment by default
RUN echo "conda activate MonoGS" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# The repository code will be mounted via volume, but we can set the entry point
CMD ["bash"]
