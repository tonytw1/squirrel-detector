import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('models/squirrelnet_ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/tflite/saved_model') # path to the SavedModel directory
tflite_model = converter.convert()

with open('models/squirrelnet_ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/tflite/squirrelnet.tflite', 'wb') as f:
	f.write(tflite_model)

print("Done")
