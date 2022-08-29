#!/usr/bin/python3

import sys
import PIL.Image
import numpy
import tensorflow as tf

image_file = sys.argv[1]
image = PIL.Image.open(image_file)

# Tflite mode needs the image to be resized for it
size = 640, 640
resized_image = image.thumbnail(size)
image = PIL.Image.open(image_file)
resized_image = image.resize(size)

# Convert our image to a numpy array
np_image = numpy.array(resized_image, dtype = numpy.float32) # This input type is different to detect.py

# Convert import image to a tensor
input_tensor = tf.convert_to_tensor(np_image)
input_tensor = input_tensor[tf.newaxis, ...]


# Load the tflite model into an interpreter
interpreter = tf.lite.Interpreter(model_path = 'models/squirrelnet_ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/tflite/squirrelnet.tflite')
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
interpreter.set_tensor(input_details[0]['index'], input_tensor)
interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
