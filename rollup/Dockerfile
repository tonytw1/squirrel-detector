FROM ubuntu

RUN apt-get update
RUN apt-get install -y ffmpeg
RUN apt-get install -y python-pip

RUN pip install boto3

WORKDIR /usr/src/app
COPY rollup/rollup.py .

CMD ["rollup.py"]
ENTRYPOINT ["python3"]
