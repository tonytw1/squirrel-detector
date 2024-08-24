FROM docker.io/tensorflow/tensorflow:2.13.0-gpu

# Dev tools
RUN apt-get update
RUN apt-get install -y git

# Dev tools required by object detection
RUN apt-get install -y protobuf-compiler
RUN apt install -y libgl1-mesa-glx
RUN DEBIAN_FRONTEND=noninteractive apt install -yq libglib2.0-0

# Clone and build the TensorFlow object detection module
RUN git clone https://github.com/tensorflow/models.git
RUN cd models/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py .
RUN cd models/research && python3 -m pip install .
