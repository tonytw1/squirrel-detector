# Where models/ conatains a check out of https://github.com/tensorflow/models.git
python3 /usr/local/lib/python3.8/dist-packages/object_detection/exporter_main_v2.py --input_type image_tensor --pipeline_config_path squirrelnet_ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8_pipeline.config --trained_checkpoint_dir ./squirrelnet_ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/ --output_directory ../models/squirrelnet_ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8

