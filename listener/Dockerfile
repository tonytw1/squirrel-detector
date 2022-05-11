FROM tensorflow/tensorflow:latest

RUN pip install paho-mqtt
RUN pip install pillow
RUN pip install requests
RUN pip install protobuf
RUN pip install opencv-python

WORKDIR /usr/src/app
COPY ./models/ ./models/
COPY listener/string_int_label_map_pb2.py .
COPY listener/listener.py .

CMD ["listener.py"]
ENTRYPOINT ["python3"]
