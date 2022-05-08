#!/usr/bin/python3

# Listens for motion detected messages on an MQTT topic
# The motion message has a JSON body:
# {
#   'image': BASE64 encoded JPEG
#   'image_file': Original path to the image file on camera device
# }
#
# Foreach motion message run TensorFlow object detection and publish the highest class scores
# {
#   'detections': {} # Map of class detection scoress
#   'image': BASE64 encoded JPEG
# }
#
# onto another MQTT detections topic
# If the detection score is high enough send a notification (email) showing the detection.

import tensorflow as tf
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
from collections import defaultdict

from google.protobuf import text_format
import string_int_label_map_pb2
from six import string_types

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')

logging.info("Importing TensorFlow")
logging.info("TensorFlow imported")

broker = os.environ.get('MOTION_MQTT_HOST')
port = int(os.environ.get('MOTION_MQTT_PORT'))
topic = os.environ.get('MOTION_MQTT_TOPIC')
detections_topic = os.environ.get('DETECTIONS_MQTT_TOPIC')

client = mqtt.Client()

last_detection = 0

label_file = os.environ.get('LABELS')

last_sent = 0

# Load the model
logging.info("Loading model")
saved_model = tf.saved_model.load(
    './models/squirrelnet_ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/saved_model/')
model = saved_model.signatures['serving_default']
logging.info("Model loaded")

# Parse a labels protobuf file into a python map
# Adapted from retrain/models/research/object_detection/utils/label_map_util.py
def load_labelmap(path):
    with open(path, "r") as labels_file:
        label_map_string = labels_file.read()
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
    logging.info("Detecting")
    prediction = model(input_tensor)
    return prediction


def on_connect(client, userdata, flags, rc):
    logging.info("Connected with result code "+str(rc))
    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe(topic)


labels = get_labels(label_file)


def send_detection_message(detections, image, annotated_image, duration, image_filename):
    message = {
        'detections': detections,
        'image': image,
        'annotated_image': annotated_image,
        'duration': duration,
        'image_filename': image_filename
    }
    detection_message = json.dumps(message)
    logging.info("Sending detection message")
    # TODO stream to mqtt
    client.publish(detections_topic, detection_message)


def send_zeros():
    global last_detection
    delta = time.time() - last_detection
    if (delta >= 30):
        zeros = {}
        for c in labels.values():
            zeros[c] = 0.0
        send_detection_message(zeros, None, None, None, None)


def annotateImage(prediction, image, image_np):
    # Given a prediction and the numpy image
    # annotate that image with a bounding box for the best prediction
    # Returns a JPEG byte array
    detection_boxes = prediction['detection_boxes'].numpy().tolist()
    detection_scores = prediction['detection_scores'].numpy().tolist()[0]
    detection_classes = prediction['detection_classes'].numpy().tolist()[0]

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

    detection_label = "{0} {1}".format(
        labels[detection_classes[0]], detection_scores[0])
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
    return img_byte_arr

def on_message(client, userdata, msg):
    global last_detection
    global last_sent
    logging.info("Message received from topic: " + msg.topic)
    message = json.loads(msg.payload)
    logging.info("Motion event found: " + message['image_file'])
    last_detection = time.time()

    # Decode the image payload
    base64_image = message['image']
    try:
        image = PIL.Image.open(BytesIO(base64.b64decode(base64_image)))
        image_filename = message['image_filename']
        image_np = numpy.array(image)

        start = time.time()
        prediction = predict_tf(image_np)
        duration = time.time() - start
        logging.info("Prediction took: {0}".format(duration))
        detection_scores = prediction['detection_scores'].numpy().tolist()[0]
        detection_classes = prediction['detection_classes'].numpy().tolist()[0]

        # Merge detection class with scores
        detected_classes = numpy.array(
            list(zip(detection_classes, detection_scores)))

        detections = defaultdict(lambda: 0)
        for i in range(0, len(detection_scores)):
            c = detection_classes[i]
            s = detection_scores[i]
            key = labels[c]
            i = detections[key]
            if s > i:
                detections[key] = s

        annotated_image_byte_arr = annotateImage(prediction, image, image_np)

        # Publish notifications for strong class detections and find the max detection strength
        base64_annotated_image = base64.b64encode(
            annotated_image_byte_arr).decode("ascii")
        send_detection_message(detections, base64_image,
                               base64_annotated_image, duration, image_filename)

        # Schedule broadcast of a non motion message
        logging.info("Scheduling send zeros")
        t = threading.Timer(30, send_zeros)
        t.start()

    except:
        println("Could not process message")

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

logging.info("Connecting to MQTT: {0} / {1}".format(broker, topic))
client.connect(broker, port, 60)
send_zeros()

client.loop_forever()
