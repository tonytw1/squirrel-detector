import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('models/squirrelnet_ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/tflite/saved_model') # path to the SavedModel directory
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# print("Writing tflife model")
TF_LITE_MODEL_PATH = 'models/squirrelnet_ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/tflite/squirrelnet.tflite'
with open(TF_LITE_MODEL_PATH, 'wb') as f:
	f.write(tflite_model)

# Convert the labels to tflite toolings format
# TODO hard dependencies; converted manually
#LABEL_PATH = 'retraining/data/tf_label_map.pbtxt'
TF_LITE_LABEL_PATH = 'models/squirrelnet_ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/tflite/squirrelnet_label_map.txt'

#print("Converting labels to label map text file")
#from object_detection.utils import label_map_util

#category_index = label_map_util.create_category_index_from_labelmap(LABEL_PATH)
#f = open(TF_LITE_LABEL_PATH, 'w')
#for class_id in range(1, 91):
#  if class_id not in category_index:
#    f.write('???\n')
#    continue
#  name = category_index[class_id]['name']
#  f.write(name+'\n')
#f.close()

# Normalise the model
print("Normalising model")
from tflite_support.metadata_writers import object_detector
from tflite_support.metadata_writers import writer_utils

writer = object_detector.MetadataWriter.create_for_inference(
    writer_utils.load_file(TF_LITE_MODEL_PATH),
    input_norm_mean=[127.5],
    input_norm_std=[127.5],
    label_file_paths=[TF_LITE_LABEL_PATH]
    )

writer_utils.save_file(writer.populate(), TF_LITE_MODEL_PATH + ".normalised")
print("Done")
