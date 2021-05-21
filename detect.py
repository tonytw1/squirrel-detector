#!/usr/bin/python3

import PIL.Image
import numpy
import tensorflow as tf

saved_model = tf.saved_model.load('models/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model/')
model = saved_model.signatures['serving_default'] #TODO This looks important and is not understood

# Convert our image to a numpy array
image = PIL.Image.open("squirrel.jpg")
np_image = numpy.array(image)

input_tensor = tf.convert_to_tensor(np_image)
input_tensor = input_tensor[tf.newaxis, ...]

detections = model(input_tensor)	# TODO doesn't match the .predict() method call in the docs

num_of_detections = int(detections['num_detections'].numpy())
for i in range(0, num_of_detections):
	detection_class = detections['detection_classes'][0][i].numpy();
	detection_score = detections['detection_scores'][0][i].numpy();
	print(str(detection_class) + ": " + str(detection_score))
