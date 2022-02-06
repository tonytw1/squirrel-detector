FROM python

RUN pip install paho-mqtt
RUN pip install pillow
RUN pip install requests

WORKDIR /usr/src/app
COPY notifications/notify.py .

CMD ["notify.py"]
ENTRYPOINT ["python3"]
