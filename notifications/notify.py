#!/usr/bin/python3

import paho.mqtt.client as mqtt

from io import BytesIO
import PIL.Image
import requests
import json
import base64
import io
import time
import os

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')

broker = os.environ.get('MOTION_MQTT_HOST')
port = int(os.environ.get('MOTION_MQTT_PORT'))
topic = os.environ.get('DETECTIONS_MQTT_TOPIC')
slack_webhook = os.environ.get('SLACK_WEBHOOK')

last_sent = 0

client = mqtt.Client()


def on_connect(client, userdata, flags, rc):
    logging.info("Connected with result code "+str(rc))
    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe(topic)


def on_message(client, userdata, msg):
    global last_detection
    global last_sent
    logging.info("Message received from topic: " + msg.topic)
    message = json.loads(msg.payload)
    detections = message['detections']

    # Find the best detection
    max = 0
    max_index = 0
    for c in detections:
        if detections[c] > max:
            max = detections[c]
            max_index = c

    if (max > 0.70) & (time.time() - last_sent > 60):
        summary = str(max_index) + ": " + str(max)

        # Decode the image payload
        base64_image = message['image']
        image = PIL.Image.open(BytesIO(base64.b64decode(base64_image)))
        # TODO images need to be uploaded to urls before slack can use them

        slack_message = json.dumps({
            'text': summary
        })
        headers = {'Content-type': 'application/json'}
        r = requests.post(slack_webhook, headers=headers, data=slack_message)
        if r.status_code == 200:
            logging.info("Slack updated: " + summary)

        # Update rate limit watermark
        last_sent = time.time()


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

logging.info("Connecting to MQTT: {0} / {1}".format(broker, topic))
client.connect(broker, port, 60)

client.loop_forever()
