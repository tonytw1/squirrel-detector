#!/usr/bin/python3
# See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
import sys
import PIL.Image
import numpy
import time

image_file = sys.argv[1]
image = PIL.Image.open(image_file)

import tensorflow as tf

# Load the tflite model into an interpreter
interpreter = tf.lite.Interpreter(model_path = 'models/squirrelnet_ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/tflite/squirrelnet-normalised.tflite')
interpreter.allocate_tensors()
print("Interpreter input details")
print(interpreter.get_input_details())

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Tflite mode needs the image to be resized for it
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
resized_image = image.resize((width, height))
input_data = numpy.expand_dims(resized_image, axis=0)

# Convert input to floating point if this model requires them
floating_model = input_details[0]['dtype'] == numpy.float32
print("Is floating model: ", floating_model)
if floating_model:
    input_data = (numpy.float32(input_data) - 127.5) / 127.5	# TODO hardcoded input_mean and input_std normalisation constants; can these be extracted from the model?

interpreter.set_tensor(input_details[0]['index'], input_data)

# Invoke
start_time = time.time()
interpreter.invoke()
stop_time = time.time()

print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))


# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])

results = numpy.squeeze(output_data)
top_k = results.argsort()[-5:][::-1]
for i in top_k:
	if floating_model:
		print('{:08.6f}: {}'.format(float(results[i]), i))
	else:
      		print('{:08.6f}: {}'.format(float(results[i] / 255.0), i))

