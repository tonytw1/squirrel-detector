# Where models/ conatains a check out of https://github.com/tensorflow/models.git
python3 models/research/object_detection/model_main_tf2.py --pipeline_config_path=training/pipeline.config --model_dir=training --alsologtostderr

