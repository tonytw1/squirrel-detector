FROM python

RUN pip install paho-mqtt==1.6.1
RUN pip install pillow
RUN pip install requests
RUN pip install numpy
RUN pip install boto3

WORKDIR /usr/src/app
COPY notifications/notify.py .

CMD ["notify.py"]
ENTRYPOINT ["python3"]
