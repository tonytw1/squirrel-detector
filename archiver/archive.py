#!/usr/bin/python3

# Listens for messages on the detections MQTT topic
# Archives the raw and the annotated images to s3

import paho.mqtt.client as mqtt

import os
import io
import json
import logging
import sys
import base64
import boto3

logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')

broker = os.environ.get('MOTION_MQTT_HOST')
port = int(os.environ.get('MOTION_MQTT_PORT'))
topic = os.environ.get('DETECTIONS_MQTT_TOPIC')

aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
s3_bucket = os.environ.get('S3_BUCKET')

s3_client = boto3.client('s3',
                         aws_access_key_id=aws_access_key_id,
                         aws_secret_access_key=aws_secret_access_key)


def on_connect(client, userdata, flags, rc):
    logging.info("Connected with result code " + str(rc))
    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe(topic)
    logging.info("Subscribed to " + topic)


def upload_image(s3_bucket, image_filename, data):
    # Upload image to private s3 bucket.
    response = s3_client.upload_fileobj(io.BytesIO(data), s3_bucket,
                                        image_filename)


def on_message(client, userdata, msg):
    global last_detection
    global last_sent
    logging.info("Message received from topic: " + msg.topic)
    message = json.loads(msg.payload)

    if 'image_filename' in message and message['image_filename'] is not None:
        image_filename = message['image_filename']
        logging.info("Received detections for image filename: " +
                     image_filename)

        if 'image' in message and message['image'] is not None:
            base64_image = message['image']
            img_bytes = base64.b64decode(base64_image)
            raw_image_path = "raw/" + image_filename
            logging.info("Saving raw image to: " + raw_image_path)
            upload_image(s3_bucket, raw_image_path, img_bytes)

        if 'annotated_image' in message and message[
                'annotated_image'] is not None:
            base64_image = message['annotated_image']
            img_bytes = base64.b64decode(base64_image)
            annotated_image_path = "annotated/" + image_filename
            logging.info("Saving annotated image to: " + annotated_image_path)
            upload_image(s3_bucket, annotated_image_path, img_bytes)


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

logging.info("Connecting to MQTT: {0} / {1}".format(broker, topic))
client.connect(broker, port, 60)

client.loop_forever()
