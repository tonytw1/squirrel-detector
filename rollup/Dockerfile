FROM ubuntu

RUN pip install boto3

RUN apt-get update
RUN apt-get install -y ffmpeg

WORKDIR /usr/src/app
COPY rollup/rollup.py .

CMD ["rollup.py"]
ENTRYPOINT ["python3"]
