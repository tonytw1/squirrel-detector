# Attempts to repeat the steps out lined in https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md
# Using the object detection Docker file or the latest tensorflow gpu image as the base resulted in version conflicts.
# Therefore we're building from base Ubuntu 20 while trying to balance the May 2021 state of the python dependencies.
# We can document our work arounds as Docker steps
FROM ubuntu:20.04

# Python 3.7 environment and dev files
RUN apt-get update
RUN apt-get install -y protobuf-compiler
RUN apt-get install -y python3
RUN apt-get install -y python3-dev
RUN apt-get install -y python3-pip
RUN python3 -m pip install -U pip

# Dev tools required by object detection
RUN apt-get install -y git
RUN apt-get install -y protobuf-compiler
RUN apt install -y libgl1-mesa-glx
RUN DEBIAN_FRONTEND=noninteractive apt install -yq libglib2.0-0

# Clone and build object detected modele
RUN git clone https://github.com/tensorflow/models.git

RUN cd models/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py .
RUN cd models/research && python3 -m pip install .

#ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 88 from C header, got 80 from PyObject
RUN python3 -m pip install pycocotools==2.0.0

# GPU
RUN echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" > /etc/apt/sources.list.d/nvidia_cuda_ubuntu2004.list
RUN apt-get install -y curl
RUN curl -sL "http://keyserver.ubuntu.com/pks/lookup?op=get&search=0xF60F4B3D7FA2AF80" | apt-key add -

RUN apt-get update
# TODO no recommended installs
RUN apt-get install -y cuda-11-0

# Could not load dynamic library 'libcusolver.so.11'; dlerror: libcusolver.so.11: cannot open shared object file: No such file or directory
RUN apt-get install -y libcusolver-11-3

# Could not load dynamic library 'libcudnn.so.8'; dlerror: libcudnn.so.8: cannot open shared object file: No such file or directory
RUN apt-get install -y libcudnn8
