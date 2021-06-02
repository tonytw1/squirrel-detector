# Where models/ conatains a check out of https://github.com/tensorflow/models.git
python3 models/research/object_detection/model_main_tf2.py --pipeline_config_path=ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8_pipeline.config --model_dir=squirrelnet --checkpoint_dir=squirrelnet --alsologtostderr

