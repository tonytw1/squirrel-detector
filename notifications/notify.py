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
import uuid

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

message_from = os.environ.get('EMAIL_FROM')
message_to = os.environ.get('EMAIL_TO')
smtp_user = os.environ.get('SMTP_USER')
smtp_password = os.environ.get('SMTP_PASSWORD')
smtp_server = os.environ.get('SMTP_HOST')

last_sent = 0

client = mqtt.Client()


def on_connect(client, userdata, flags, rc):
    logging.info("Connected with result code "+str(rc))
    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe(topic)


def send_slack(summary):
    slack_message = json.dumps({
        'text': summary
    })
    headers = {'Content-type': 'application/json'}
    r = requests.post(slack_webhook, headers=headers, data=slack_message)
    if r.status_code == 200:
        logging.info("Slack updated: " + summary)

def send_email(summary, detections, image_filename, img_byte_arr, duration):
    # Send an email notification for this event
    m = ""
    for c in detections:
        label_display_name = c
        detection_line = label_display_name + ": " + str(detections[c])
        m = m + detection_line

    subject = summary
    cid = image_filename + uuid.uuid4().hex

    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = message_from
    message["To"] = message_to

    text = """\
Motion detected
        """
    html = """\
<p><pre>{0}</pre></p>
<p>Took: {2}</p>
<p><img src="cid:{1}"></p>
<hr/>
        """
    plain_part = MIMEText(text, "plain")
    html_part = MIMEText(html.format(m, cid, duration), "html")

    image_attachment = MIMEImage(img_byte_arr, _subtype="jpeg")
    image_attachment.add_header(
        'Content-Disposition', 'attachment; filename={0}'.format(image_filename))
    image_attachment.add_header('Content-ID', '<{}>'.format(cid))

    message.attach(plain_part)
    message.attach(html_part)
    message.attach(image_attachment)

    server = smtplib.SMTP(smtp_server, 587)
    server.login(smtp_user, smtp_password)
    server.sendmail(message_from, message_to, message.as_string())
    logging.info("Sent notification: " + subject)

def on_message(client, userdata, msg):
    global last_detection
    global last_sent
    logging.info("Message received from topic: " + msg.topic)
    message = json.loads(msg.payload)
    detections = message['detections']
    duration = message['duration']

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
        img_bytes = base64.b64decode(base64_image)
        image = PIL.Image.open(BytesIO(img_bytes))
        # TODO images need to be uploaded to urls before slack can use them

        send_slack(summary)
        send_email(summary, detections, "TODO", img_bytes, duration)

        # Update rate limit watermark
        last_sent = time.time()


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

logging.info("Connecting to MQTT: {0} / {1}".format(broker, topic))
client.connect(broker, port, 60)

client.loop_forever()

