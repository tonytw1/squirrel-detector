#!/usr/bin/python3

import sys 
import PIL.Image
import numpy
import time

image_file = sys.argv[1]
# Convert our image to a numpy array
image = PIL.Image.open(image_file)
np_image = numpy.array(image)

print("Importing TensorFlow")
import tensorflow as tf

print("Loading saved model")
saved_model = tf.saved_model.load('models/squirrelnet_ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model')
model = saved_model.signatures['serving_default'] #TODO This looks important and is not understood

input_tensor = tf.convert_to_tensor(np_image)
input_tensor = input_tensor[tf.newaxis, ...]

detections = model(input_tensor)

print("Detecting")
start_time = time.time()
detections = model(input_tensor)        # TODO doesn't match the .predict() method call in the docs
stop_time = time.time()
print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))

num = int(detections['num_detections'].numpy())
for i in range(0, num):
	detection_class = detections['detection_classes'][0][i].numpy();
	detection_score = detections['detection_scores'][0][i].numpy();
	print(str(detection_class) + ": " + str(detection_score))
