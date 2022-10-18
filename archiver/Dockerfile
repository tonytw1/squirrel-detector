FROM python

RUN pip install paho-mqtt
RUN pip install boto3

WORKDIR /usr/src/app
COPY archiver/archive.py .

CMD ["archive.py"]
ENTRYPOINT ["python3"]
