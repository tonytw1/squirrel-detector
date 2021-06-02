FROM python:3

RUN apt-get update
RUN apt-get install --no-install-recommends -y python3-opencv

RUN pip install paho-mqtt
RUN pip install pillow
RUN pip install numpy
RUN pip install requests
RUN pip install protobuf

WORKDIR /usr/src/app
COPY string_int_label_map_pb2.py .
COPY squirrelnet_label_map.pbtxt . 
COPY listener.py .
CMD ["listener.py"]
ENTRYPOINT ["python3"]
