# Where models/ conatains a check out of https://github.com/tensorflow/models.git
# and pretrained_models/ contains and extract of http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz
python3 /models/research/object_detection/model_main_tf2.py --pipeline_config_path=squirrelnet_ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8_pipeline.config --model_dir=squirrelnet_ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8 --alsologtostderr

