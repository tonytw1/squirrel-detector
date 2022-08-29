# Attempts to repeat the steps out lined in https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md
FROM eelpie:ubuntu-nvidia-base-image

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

# Clone and build the TensorFlow object detection module
RUN git clone https://github.com/tensorflow/models.git
RUN cd models/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py .
RUN cd models/research && python3 -m pip install .

RUN pip install tflite-support
