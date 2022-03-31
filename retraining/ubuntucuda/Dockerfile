# Base Ubuntu 20 with Nivida drivers and Cuda installed
FROM ubuntu:20.04

# Update and bootstrap utilities needed by nvidia packages
RUN apt-get update
RUN apt-get install -y apt-utils wget gnupg2 software-properties-common
RUN apt-get install -y tzdata

# Nivida GPU drivers and Cuda tools
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"

RUN apt-get update
# TODO do not need to install recommended
RUN apt-get -y install cuda
RUN apt-get install -y libcudnn8
