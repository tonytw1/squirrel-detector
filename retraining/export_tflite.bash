# Where models/ conatains a check out of https://github.com/tensorflow/models.git
python3 /models/research/object_detection/export_tflite_graph_tf2.py --pipeline_config_path squirrelnet_pipeline.config --trained_checkpoint_dir ./squirrelnet/ --output_directory ../models/squirrelnet_ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/tflite
