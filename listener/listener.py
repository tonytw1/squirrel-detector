#!/usr/bin/python3

# Listens for motion messages on the motion MQTT topic
#
# The motion messages have this JSON body:
# {
#   'image': BASE64 encoded JPEG
#   'image_filename': Filename of the original image file on camera device
#   'motion': Optional bounding box for motion area which triggered this event
# }
#
# Foreach motion message run TensorFlow object detection and publish a message onto the detections MQTT topic
# {
#   'detections': Map of class detection scores
#   'image': BASE64 encoded original JPEG image
#   'annotated_image': BASE64 encoded annotated JPEG image,
#   'duration': Detection duration in ms,
#   'image_filename': Image filename
# }

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

from colour import Color

logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
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
    './models/squirrelnet_ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model/'
)
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
    logging.info("Connected with result code " + str(rc))
    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe(topic)


labels = get_labels(label_file)
colours = list(Color("red").range_to(Color("blue"), len(labels.values())))


def send_detection_message(detections, image, annotated_image, duration,
                           image_filename):
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


def annotateImage(prediction, image, image_np, motion):
    # Given a prediction and the numpy image
    # annotate that image with a bounding box for each confident prediction
    # Returns a JPEG byte array

    width, height = image.size

    # Annotate with motion box if available
    image_with_motion = image_np
    if motion:
        x1 = motion[0]
        y1 = motion[1]
        x2 = motion[2]
        y2 = motion[3]

        white = (255, 255, 255)
        image_with_motion = cv2.rectangle(image_with_motion, (x1, y1),
                                          (x2, y2), white, 1)

    # Annotate with detection boxes
    image_with_detections = image_with_motion

    # Reverse to paint the most confident prediction on top of the less confident ones
    detection_boxes = list(
        reversed(prediction['detection_boxes'].numpy().tolist()[0]))
    detection_scores = list(
        reversed(prediction['detection_scores'].numpy().tolist()[0]))
    detection_classes = list(
        reversed(prediction['detection_classes'].numpy().tolist()[0]))

    for i in range(0, len(detection_boxes)):
        detection_box = detection_boxes[i]
        detection_class = detection_classes[i]
        detection_score = detection_scores[i]

        if (detection_score > 0.90):
            y1 = int(height * detection_box[0])
            x1 = int(width * detection_box[1])
            y2 = int(height * detection_box[2])
            x2 = int(width * detection_box[3])

            x = int(detection_class) - 1
            colour = colours[x]
            rgb_colour = list(map(lambda i: int(i) * 255, colour.rgb))

            image_with_detections = cv2.rectangle(image_with_detections,
                                                  (x1, y1), (x2, y2),
                                                  rgb_colour, 3)

            detection_label = "{0} {1}".format(labels[detection_class],
                                               round(detection_score, 4))
            label_padding = 5
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image_with_detections, detection_label,
                        (x1, y1 - label_padding), font, 0.5, rgb_colour, 1,
                        cv2.LINE_AA)

    # Convert back to A PIL image for output to jpeg
    image_with_detections_pil = PIL.Image.fromarray(image_with_detections)
    img_byte_arr = io.BytesIO()
    image_with_detections_pil.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr


def on_message(client, userdata, msg):
    global last_detection
    global last_sent

    try:
        logging.info("Message received from topic: " + msg.topic)
        message = json.loads(msg.payload)
        image_filename = message['image_filename']

        logging.info("Motion event found: " + image_filename)
        last_detection = time.time()

        # Decode the image payload
        base64_image = message['image']

        image = PIL.Image.open(BytesIO(base64.b64decode(base64_image)))
        image_np = numpy.array(image)

        start = time.time()
        prediction = predict_tf(image_np)
        duration = time.time() - start
        logging.info("Prediction took: {0}".format(duration))
        detection_classes = prediction['detection_classes'].numpy().tolist()[0]
        detection_scores = prediction['detection_scores'].numpy().tolist()[0]

        detections = defaultdict(lambda: 0)
        for i in range(0, len(detection_scores)):
            c = detection_classes[i]
            s = detection_scores[i]
            key = labels[c]
            i = detections[key]
            if s > i:
                detections[key] = s

        motion = None
        if 'motion' in message:
            motion = message['motion']
            motion_center_x = int(motion['bounding_box_center_x'])
            motion_center_y = int(motion['bounding_box_center_y'])
            motion_width = int(motion['bounding_box_width'])
            motion_height = int(motion['bounding_box_height'])

            motion_x1 = int(motion_center_x - (motion_width / 2))
            motion_x2 = int(motion_center_x + (motion_width / 2))
            motion_y1 = int(motion_center_y - (motion_height / 2))
            motion_y2 = int(motion_center_y + (motion_height / 2))

            motion = [motion_x1, motion_y1, motion_x2, motion_y2]

        annotated_image_byte_arr = annotateImage(prediction, image, image_np,
                                                 motion)

        # Publish notifications for strong class detections and find the max detection strength
        base64_annotated_image = base64.b64encode(
            annotated_image_byte_arr).decode("ascii")
        send_detection_message(detections, base64_image,
                               base64_annotated_image, duration,
                               image_filename)

        # Schedule broadcast of a non motion message
        logging.info("Scheduling send zeros")
        t = threading.Timer(30, send_zeros)
        t.start()

    except Exception as e:
        logging.exception("Could not process message", exc_info=True)


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

logging.info("Connecting to MQTT: {0} / {1}".format(broker, topic))
client.connect(broker, port, 60)
send_zeros()

client.loop_forever()
