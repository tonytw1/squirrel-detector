import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('/Users/tony/git/squirrel-detector/models/squirrelnet_ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/saved_model') # path to the SavedModel directory
tflite_model = converter.convert()

println("Done")