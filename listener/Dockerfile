FROM tensorflow/tensorflow:2.21.0-gpu

RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Install everything in one resolve so pip keeps TensorFlow, its CUDA libs and protobuf compatible
RUN pip install \
      "tensorflow[and-cuda]==2.21.0" \
      "nvidia-cudnn-cu12==9.3.0.75" \
      paho-mqtt==1.6.1 \
      pillow \
      requests \
      protobuf \
      opencv-python-headless \
      colour

WORKDIR /usr/src/app
COPY ./models/ ./models/
COPY listener/string_int_label_map_pb2.py .
COPY listener/listener.py .

ENTRYPOINT ["python3"]
CMD ["listener.py"]

