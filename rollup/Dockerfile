FROM ubuntu

RUN apt-get update
RUN apt-get install -y ffmpeg

RUN apt install python3.10
RUN apt-get install -y python3-pip

RUN pip3 install boto3

WORKDIR /usr/src/app
COPY rollup/rollup.py .

CMD ["rollup.py"]
ENTRYPOINT ["python3"]
