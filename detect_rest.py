#!/usr/bin/python3

import sys
import PIL.Image
import numpy
import requests

image_file = sys.argv[1]

image = PIL.Image.open(image_file)
image_np = numpy.array(image)

payload = {"instances": [image_np.tolist()]}

res = requests.post("http://localhost:8501/v1/models/ssd_mobilenet_v2_320x320_coco17_tpu-8:predict", json=payload)

json = res.json();
predictions = json['predictions']

for prediction in predictions:
	num_detections = prediction['num_detections']
	print("Number of detections: " + str(num_detections));

	detection_scores = prediction['detection_scores'];
	for score in detection_scores[:3]:
		print(score)

	detection_classes = prediction['detection_classes']
	for detection_class in detection_classes[:3]:
		print(detection_class)

	print(len(detection_scores))
	print(len(detection_classes))

	print(max(detection_scores))
	print(max(detection_classes))

	for prediction_field in prediction.keys():
		print(prediction_field)

