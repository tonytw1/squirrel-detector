#!/usr/bin/python3

import paho.mqtt.client as mqtt

from io import BytesIO
import PIL.Image
import numpy
import requests
import json
import base64

broker = '192.168.1.27'
topic = 'motion'

tensorflowserver_url = "http://localhost:32701/v1"
model = "ssd_mobilenet_v2_320x320_coco17_tpu-8"

# Given a PIL image call TensorFlow Serving to detect objects
def predict(image):
	image_np = numpy.array(image)
	payload = {"instances": [image_np.tolist()]}

	predict_url = tensorflowserver_url + "/models/" + model + ":predict"
	res = requests.post(predict_url, json=payload)

	json = res.json();
	predictions = json['predictions']
	return predictions


def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe(topic)

def on_message(client, userdata, msg):
    print("Message recieved from topic: " + msg.topic)
    message = json.loads(msg.payload)
    print("Motion event found: " + message['image_file'])

    # Decode the image payload
    image = PIL.Image.open(BytesIO(base64.b64decode(message['image'])))
    predictions = predict(image)

    print("Got predicitions: ")
    num = len(predictions)
    print(num)

    prediction = predictions[0];
    num = len(prediction)
    detection_scores = prediction['detection_scores'];
    detection_classes = prediction['detection_classes']
    print(num)
    for i in range(num):
        print(str(i) + ": " + str(detection_classes[i]) + ": " + str(detection_scores[i]))

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect(broker, 1883, 60)

client.loop_forever()
