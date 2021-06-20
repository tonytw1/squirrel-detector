#!/usr/bin/python3

# Listens for motion detected messages on an MQTT topic
# Foreach motion message run TensorFlow object detection and publish the highest class scores 
# onto another MQTT detections topic
# If the detection score is high enough send a notification (email) showing the detection.

import paho.mqtt.client as mqtt

from io import BytesIO
import PIL.Image
import numpy
import requests
import json
import base64
import io
import time
import os
import cv2
import uuid
import time
import threading

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

from google.protobuf import text_format
import string_int_label_map_pb2
from six import string_types

print("Importing TensorFlow")
import tensorflow as tf

broker = os.environ.get('MOTION_MQTT_HOST')
topic = os.environ.get('MOTION_MQTT_TOPIC')
detections_topic = os.environ.get('DETECTIONS_MQTT_TOPIC')

last_detection = 0

label_file = os.environ.get('LABELS')

message_from = os.environ.get('EMAIL_FROM')
message_to = os.environ.get('EMAIL_TO')

smtp_user = os.environ.get('SMTP_USER')
smtp_password = os.environ.get('SMTP_PASSWORD')
smtp_server = os.environ.get('SMTP_HOST')

last_sent = 0

# Load the model
print("Loading model")
saved_model = tf.saved_model.load('./models/squirrelnet_ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/saved_model/')
model = saved_model.signatures['serving_default']
print("Model loaded")

# Parse a labels protobuf file into a python map
# Adapted from retrain/models/research/object_detection/utils/label_map_util.py
def load_labelmap(path):
  with open (path, "r") as labels_file:
    label_map_string=labels_file.read()
    label_map = string_int_label_map_pb2.StringIntLabelMap()
    try:
      text_format.Merge(label_map_string, label_map)
    except text_format.ParseError:
      label_map.ParseFromString(label_map_string)
  return label_map

def get_labels(label_map_path):
  label_map = load_labelmap(label_map_path)
  label_map_dict = {}
  for item in label_map.item:
      label_map_dict[item.id] = item.name
  return label_map_dict

# Given a PIL image call TensorFlow to detect objects
def predict_tf(image_np):
	input_tensor = tf.convert_to_tensor(image_np)
	input_tensor = input_tensor[tf.newaxis, ...]
	print("Detecting")
	prediction = model(input_tensor)
	return prediction

def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe(topic)

labels = get_labels(label_file)

def send_zeros(client):
    print("Considering Sending zero")	
    if (time.time() - last_detection >= 30): 
        print("Really sending zeros")

def on_message(client, userdata, msg):
    global last_sent
    print("Message recieved from topic: " + msg.topic)
    message = json.loads(msg.payload)
    print("Motion event found: " + message['image_file'])
    last_detection = time.time()

    # Decode the image payload
    base64_image = message['image']
    image = PIL.Image.open(BytesIO(base64.b64decode(base64_image)))
    image_filename =  message['image_file'].rsplit('/', 1)[1]

    image_np = numpy.array(image)

    start = time.time()
    prediction = predict_tf(image_np)
    duration = time.time() - start
    print("Prediction took: {0}".format(duration))
    detection_scores = prediction['detection_scores'].numpy().tolist()[0]
    detection_classes = prediction['detection_classes'].numpy().tolist()[0]

    # Merge detection class with scores
    detected_classes = numpy.array(list(zip(detection_classes, detection_scores)))

    from collections import defaultdict

    maxes = defaultdict(lambda: 0)
    for i in range(0, len(detection_scores)):
        c = detection_classes[i]
        s = detection_scores[i]
        i = maxes[c]
        if s > i:
           maxes[c] = s

    # Publish notifications for strong class detections and find the max detection strength
    max = 0
    max_index = 0
    for c in maxes:
        label_display_name = labels[c]
        detection_message = label_display_name + ":" + str(maxes[c])
        print(detection_message)
        client.publish(detections_topic, detection_message)
        if maxes[c] > max:
            max = maxes[c]
            max_index = c

    # Schedule broadcast of a non motion message 
    t = threading.Timer(10, send_zeros(client))
    t.start() 
  
    if (max > 0.80) & (time.time() - last_sent > 60):
        # Send an email notification for this event
        # Slack would probably be more immediate 
        m = "";
        for c in maxes:
            label_display_name = labels[c]
            detection_line = label_display_name + ": " + str(maxes[c])
            m = m + detection_line

        subject = 'Motion detected - {0} {1}'.format(labels[max_index], maxes[max_index])
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

        detection_boxes = prediction['detection_boxes'].numpy().tolist()
        detection_box = detection_boxes[0][0]

        image_with_detections = image_np

        width, height = image.size
        green = (0, 255, 0)

        y1 = int(height * detection_box[0])
        x1 = int(width * detection_box[1])
        y2 = int(height * detection_box[2])
        x2 = int(width * detection_box[3])
        image_with_detections = cv2.rectangle(
            image_with_detections,
            (x1, y1),
            (x2, y2),
            green,
            3
        )

        detection_label = "{0} {1}".format(labels[detection_classes[0]], detection_scores[0])
        label_padding = 5
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            image_with_detections,
            detection_label,
            (x1, y1 - label_padding),
            font,
            0.5,
            green,
            1,
            cv2.LINE_AA
        )

        image_with_detections_pil = PIL.Image.fromarray(image_with_detections)

        img_byte_arr = io.BytesIO()
        image_with_detections_pil.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()


        image_attachment = MIMEImage(img_byte_arr, _subtype="jpeg")
        image_attachment.add_header('Content-Disposition', 'attachment; filename={0}'.format(image_filename))
        image_attachment.add_header('Content-ID', '<{}>'.format(cid))

        message.attach(plain_part)
        message.attach(html_part)
        message.attach(image_attachment)

        server = smtplib.SMTP(smtp_server, 587)
        server.login(smtp_user, smtp_password)
        server.sendmail(message_from, message_to, message.as_string())
        print("Sent notification: " + subject)

        # Update rate limit watermark
        last_sent = time.time()

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

print("Connecting to MQTT: {0} / {1}".format(broker, topic))
client.connect(broker, 1883, 60)

client.loop_forever()

