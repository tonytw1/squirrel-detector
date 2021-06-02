# Where models/ conatains a check out of https://github.com/tensorflow/models.git
python3 models/research/object_detection/exporter_main_v2.py --input_type image_tensor --pipeline_config_path  ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8_pipeline.config --trained_checkpoint_dir ./squirrelnet/ --output_directory exported/squirrelnet_ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8

