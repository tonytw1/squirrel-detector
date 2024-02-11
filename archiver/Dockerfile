FROM python

RUN pip install paho-mqtt==1.6.1
RUN pip install boto3

WORKDIR /usr/src/app
COPY archiver/archive.py .

CMD ["archive.py"]
ENTRYPOINT ["python3"]
