# Base Ubuntu 20 with Nivida drivers and Cuda installed
FROM ubuntu:20.04

# Update and bootstrap utilities needed by nvidia packages
RUN apt-get update
RUN apt-get install -y apt-utils wget gnupg2 software-properties-common nano
RUN apt-get install -y tzdata

# Nivida GPU drivers and Cuda tools
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb

RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
RUN apt-get update

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y keyboard-configuration
RUN apt-get install --no-install-recommends -y cuda 
RUN apt-get install --no-install-recommends -y libcudnn8
