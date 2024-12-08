import sys
import numpy
import PIL.Image
from datetime import datetime

image_file = sys.argv[1]

print("Importing TensorFlow")
import tensorflow as tf
print("Finished importing TensorFlow")

print("Loading model")
saved_model = tf.saved_model.load('models/squirrelnet_ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model')
model = saved_model.signatures['serving_default']
print("Finished loading model")


for i in range(1, len(sys.argv)):
	start = datetime.now()
	image_file = sys.argv[i]
	print(f"Loading image {image_file}")
	image = PIL.Image.open(image_file)
	np_image = numpy.array(image)

	print(f"Converting {image_file} to input tensor")
	input_tensor = tf.convert_to_tensor(np_image)
	input_tensor = input_tensor[tf.newaxis, ...]

	print(f"Detecting in {image_file}")
	detection_start = datetime.now()
	detections = model(input_tensor)
	detection_end = datetime.now()
	detection_duration = (detection_end.timestamp() - detection_start.timestamp()) * 1000
	print(f"Detection completed for {image_file} in {detection_duration}ms")
	end = datetime.now()
	num = int(detections['num_detections'].numpy())
	for i in range(0, num):
		detection_class = detections['detection_classes'][0][i].numpy()
		detection_score = detections['detection_scores'][0][i].numpy()
		print(str(detection_class) + ": " + str(detection_score))

	end_to_end_duration = (end.timestamp() - start.timestamp()) * 1000
	print(f"{image_file} took {end_to_end_duration}ms end to end")
