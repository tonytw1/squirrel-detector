# Where models/ conatains a check out of https://github.com/tensorflow/models.git
python3 models/research/object_detection/exporter_main_v2.py --input_type image_tensor --pipeline_config_path ./training/pipeline.config --trained_checkpoint_dir ./training/ --output_directory exported/squirrelnet
