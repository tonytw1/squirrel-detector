FROM python:3
RUN pip install paho-mqtt
RUN pip install pillow
RUN pip install numpy
RUN pip install requests
RUN pip install protobuf
RUN pip install opencv-python

WORKDIR /usr/src/app
COPY string_int_label_map_pb2.py .
COPY squirrelnet_label_map.pbtxt . 
COPY listener.py .
CMD ["listener.py"]
ENTRYPOINT ["python3"]
