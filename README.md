# Squirrel detector

![Mr Squirrel](images/squirrel.jpg)

- [TLDR](#tldr)
- [Hardware](#hardware)
- [Detecting motion and capturing images](#detecting-motion-and-capturing-images)
- [Not squirrel](https://github.com/tonytw1/squirrel-detector#not-squirrel)
- [Object detection APIs](#object-detection-apis)
- [TensorFlow object detection models](#tensorflow-object-detection-models)
- [Testing in Google Colab](#testing-in-google-colab)
- [Running a model with TensorFlow Serving](#running-a-model-with-tensorflow-serving)

- [Retraining](#retraining)
- [Annotating images](#annotating-images)
- [Training](#training)
- [Putting it all together](#putting-it-all-together)

## TLDR

During lock down we were adopted by the squirrel which frequents our garden.

Now we'd like to be notified when a squirrel is outside the window, so
we built a Raspberry Pi webcam and trained a TensorFlow model to recognise squirrels.


## Hardware

We're using a [Raspberry Pi Zero W](https://www.raspberrypi.org/products/raspberry-pi-zero-w/) with the
[Camera Module V2](https://www.raspberrypi.org/products/camera-module-v2/).

This gives us Wifi, 1 core and 512Mb of memory.

Thdule appears as a Video4Linux device.
You can see device details with this command:
```
v4l2-ctl --all
```

## Detecting motion and capturing images
[Motion](https://motion-project.github.io) is available as a Raspberry Pi package.
Ite camera mo does a good job of detecting movement and creating image files and bounding boxes.

Here's an example of Motion detecting and bounding a movement:
![This is not a squirrel](images/not_squirrel.jpg)

We'd like Motion to detect bounding boxes but not draw them on the saved image files.
This configuration line seems todo this:
```
locate_motion_mode preview
```

Motion seems to have an issue with the Pi camera's auto exposure mode.
This can be worked around by setting the camera to 1024 x 640.

Our camera is connected to a small device with limited processing capability.
We want to send the image somewhere where a more capable machine can look at it.

Let's use a python script to catch the Motion events and publish them.

This script needs to capture the image file path and the bounding box from Motion and encode them into a message.

The Motion `on_picture_save` event is able to give us the image file path and bounding box.
We can hook these together with this configuration line:

`on_picture_save python3 /home/pi/on_motion_detected.py %f %w %h %K %L %i %J`

We'll need to encode the image file for inclusion in a message.
Base64 encoding should be enough.

We'll use MQTT as the message format. MQTT is really practical about message sizes limits.
We can publish the motion messages to a MQTT topic which other machines can subscribe to.

This happens in the script `on_motion_detected.py`.

![on_motion_detected.py](on_motion_detected.py)


## Not squirrel

It quickly became apparent that there were more than squirrels in the garden.

![Not squirrel](images/fox.jpg)

This is not a squirrel.

We'll want to categorise the objects in the motion messages so that we can filter for squirrels.

We have a message containing a still image with a bounding box enclosing an area of motion.
We want to phrase this and maybe crop to the bounding box.
We can then send the image to an object detection API.


## Object detection APIs

How are we going to classify the moving objects Motion has detected?
We'll like to pass the area of interest to an object detection API.


### Google Vision

Google Vision seems to be the gold standard for object detection and has a nice python API.
It also seems to know about squirrels.

Here's a script to detect objects an image file and it's sample output.

[google-vision.py](google-vision.py)

![Google Vision output](google_vision.png)


### Local alternatives?

The free tier for Google Vision isn't suited for continuous use.
Is there anything we can run locally?

Pretrained TensorFlow object detection models are available and running one locally might be an interesting side quest.

There are 2 interesting problems here. Can we find a model which can detect the objects we're interested in and
can we easily run it and call it locally.

The model will need to be wrapped in some sort of API so we can call it from our message handling script.


## TensorFlow object detection models

The [TensorFlow Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
contains saved models have been pretrained and can be downloaded.

They can be used to run predictions against our images.

Lets pick a pretrained model and try to run it against one of our test images.

Working on a local machine I was blocked almost immediately with an error while trying to load the saved model.

This could be a mismatch between TensorFlow 2.5 and the available examples.

Retreating to [https://colab.research.google.com](Google Colab) notebooks offered a known good development environment to get started in.


### Testing in Google Colab

Alot of data development work goes on in notebook environments like Jupyter and [https://colab.research.google.com](Google Colab). The data community have discovered a really
interesting way of working here. I'd encourage an software developer who haven't seen this before to have a look.

With a saved model imported into our Colab notebook we can load one of our test images and ask the model to predict the visible objects.

![Colan prediction](colab.png)

Requesting a prediction.

![Model predict](predict.png)

The prediction returns a large map of results.

![Prediction results](predictions.png)

`detection_classes` and `detection_scores` are interesting.
This turns out to mean a 73% confidence of a class 17 object.

What are the classes and why are the values all below 100?

The saved model was trained on the COCO image set.
It was only taught about 90 unique objects and the class ids refer to one of them.

The COCO labels are available in the file `mscoco_label_map.pbtxt`.

```
item {
  name: "/m/015qbp"
  id: 14
  display_name: "parking meter"
}
item {
  name: "/m/01yrx"
  id: 17
  display_name: "cat"
}
item {
  name: "/m/04dr76w"
  id: 44
  display_name: "bottle"
}
```


### Not cat

Plotting the most confident prediction over the image:
![Not cat](not_cat.png)

Looking up class 17 in the label file we find `cat`.

Close but not quite right. It looks like the model doesn't know about squirrels!

Looking in the COCO labels file confirms that squirrels are not one of the classes this model was trained on.

### Local detection script

Back porting what we learnt in the Colab worksheet we can create a local script which can make the same prediction.
There is plenty in there which I don't yet understand yet.

[detect.py](detect.py)


### Resolving labels

TensorFlow gives the impression that resolving class ids into labels is not it's concern.
We'll need to spike out a way to user the labels file to resolve readable name for classes in the predictions returned from TensorFlow.

[labels/labels.py](labels/labels.py)


### Premature productionising

We've now verified that we can use TensorFlow to run a pretrained model locally.

That pretrained model doesn't know about the specific animals we're interested in but can probably be retrained.

Let's move onto productionising what we have on the assumption we'll be able to improve the model later.



## Running a model with TensorFlow Serving

[https://www.tensorflow.org/tfx/serving/docker](TensorFlow Serving) makes it easy to deploy new algorithms and experiments, while keeping the same server architecture and APIs.

This looks exactly what we want. A way to deploy a saved model behind a REST API.

A Docker container is provided. 
We use this as a base image to produce an image with our model baked into it.

[serving/Dockerfile](serving/Dockerfile)

Testing locally:
`
docker run -p 8501:8501 -e MODEL_NAME=ssd_mobilenet_v2_320x320_coco17_tpu-8 eelpie/tensorflowserving
`

Check the models is available at `http://localhost:8501/v1/models/ssd_mobilenet_v2_320x320_coco17_tpu-8`

```
{
    "model_version_status": [
        {
            "version": "1",
            "state": "AVAILABLE",
            "status": {
                "error_code": "OK",
                "error_message": ""
            }
        }
    ]
}
```
Now we can ask for a prediction with an HTTP call rather than importing the TensorFlow model into our script.

[detect_rest.py](detect_rest.py)


## Retraining

Our downloaded model doesn't know about squirrels. We need to retrain it.


### Collecting training data

Unlike humans animals won't generally give out personally identifying informational for free.
They will trade it for nuts though.

Collecting several day's images gave a collection of several hundred training images with examples of most of the garden animals.


### Annotating images

TensorFlow wants a set of custom classes representing the objects we are interested in (ie. squirrel, fox etc).
It also needs a set of example images with instances of these classes highlighted.
If the question is where is the squirrel in this picture then we need to provide many examples of the correct answer.

An image annotation tool like [VoTT (Visual Object Tagging Tool)](https://github.com/microsoft/VoTT) will help here.

Tagging with VoTT:
![Squirrel tagged in VoTT](vott-squirrel.png)
![Fox tagged in VoTT](vott-fox.png)

These tools are optimised for smooth workflow.
I managed to tag 230 images in 30 minutes on my first attempt.
This was much quicker than expected and a somewhat cathartic.

VoTT can export to the TensorFlow Records format for direct import into TensorFlow.



### Split the data

We need to reserve some of our test data for testing of our trained model.
Just like a real exam it needs to be tested on questions it's not seen before.

```
cd Squirrels-TFRecords-export
mkdir training
mkdir eval
mv *.tfrecord training
mv training/7* eval
```

A better splitting would probably try to get a representative spread of classes into the evaluation folder.


### Object Detection API

The Object Detection API seems to be TensorFlow's high level wrapper around this type of problem.

The [https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md](installation instructions) seem to have suffered from python and CUDA dependency rot.

Getting a working GPU enabled install of TensorFlow and the Object Detection API was difficult.

My attempts to working around this are documented in [retraining/Dockerfile](retraining/Dockerfile).


### Training

With our test data prepared, we need to add a pretrained existing model,
a training pipeline to describe the training task and a checkpoint to describe the training the existing model has already undergone.

We start the training process and over the course of several hours TensorFlow will attempt to discover the model parameters
which minimise prediction error (or loss) against our training images.

[retraining/train.bash](retraining/train.bash)

The loss value would be expected to tread downwards during training; probably towards a value between 0.0 and 1.0.

![Loss converging](loss-converging.png)



While training TensorFlow will periodially log out a progress report.

```
I0524 17:03:34.777422 139797117065024 model_lib_v2.py:680] Step 400 per-step time 1.050s loss=1.978
```

For a constant batch size the per-step-time should give is a rough way to compare different hardware options.

Comparing some of the locally available hardware:

4 core 3.4 GHz CPU ~ 5.0s
2 x 10 core 2.8 CPU ~ 2.7s
GTX 1050 Ti 4Gb ~ 1.0s


### Checkpoints

As it trains TensorFlow periodically drops check points.
These represent the current parameter settings for the model.
Training is about finding the model parameters which best fit the data.

![Check points](checkpoints.png)

Check points can be used to port pause and resume training.
They can also be used to resume training on a faster GPU enabled cloud instance.



### Evaluating while training

In theory we can use the evaluation images we reserved to continually evaluate the models accuracy as it trains.

```
export CUDA_VISIBLE_DEVICES=-1
python3 models/research/object_detection/model_main_tf2.py --pipeline_config_path=training/pipeline.config --model_dir=training --checkpoint_dir=training  --alsologtostderr
```

Note how we have to disable CUDA to prevent the training and evaluation processes competing for the GPU.

This fails. Could be a TensorFlow 2.5 incompatibility.
```
INFO:tensorflow:Performing evaluation on 76 images.
I0527 09:17:59.674201 139992811644736 coco_evaluation.py:293] Performing evaluation on 76 images.
creating index...
index created!
INFO:tensorflow:Loading and preparing annotation results...
I0527 09:17:59.675634 139992811644736 coco_tools.py:116] Loading and preparing annotation results...
INFO:tensorflow:DONE (t=0.00s)
I0527 09:17:59.680794 139992811644736 coco_tools.py:138] DONE (t=0.00s)
creating index...
index created!
Traceback (most recent call last):
  File "models/research/object_detection/model_main_tf2.py", line 114, in <module>
    tf.compat.v1.app.run()
  File "/usr/local/lib/python3.8/dist-packages/tensorflow/python/platform/app.py", line 40, in run
    _run(main=main, argv=argv, flags_parser=_parse_flags_tolerate_undef)
  File "/usr/local/lib/python3.8/dist-packages/absl/app.py", line 303, in run
    _run_main(main, args)
  File "/usr/local/lib/python3.8/dist-packages/absl/app.py", line 251, in _run_main
    sys.exit(main(argv))
  File "models/research/object_detection/model_main_tf2.py", line 81, in main
    model_lib_v2.eval_continuously(
  File "/usr/local/lib/python3.8/dist-packages/object_detection/model_lib_v2.py", line 1133, in eval_continuously
    eager_eval_loop(
  File "/usr/local/lib/python3.8/dist-packages/object_detection/model_lib_v2.py", line 984, in eager_eval_loop
    eval_metrics.update(evaluator.evaluate())
  File "/usr/local/lib/python3.8/dist-packages/object_detection/metrics/coco_evaluation.py", line 302, in evaluate
    box_evaluator = coco_tools.COCOEvalWrapper(
  File "/usr/local/lib/python3.8/dist-packages/object_detection/metrics/coco_tools.py", line 207, in __init__
    cocoeval.COCOeval.__init__(self, groundtruth, detections, iouType=iou_type)
  File "/usr/local/lib/python3.8/dist-packages/pycocotools/cocoeval.py", line 76, in __init__
    self.params = Params(iouType=iouType) # parameters
  File "/usr/local/lib/python3.8/dist-packages/pycocotools/cocoeval.py", line 527, in __init__
    self.setDetParams()
  File "/usr/local/lib/python3.8/dist-packages/pycocotools/cocoeval.py", line 507, in setDetParams
    self.iouThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
  File "<__array_function__ internals>", line 5, in linspace
  File "/usr/local/lib/python3.8/dist-packages/numpy/core/function_base.py", line 113, in linspace
    num = operator.index(num)
TypeError: 'numpy.float64' object cannot be interpreted as an integer
```



### Loss blow outs

Occasionally it would explode like this:

```
INFO:tensorflow:Step 11300 per-step time 0.695s loss=0.727
INFO:tensorflow:Step 11400 per-step time 0.703s loss=0.633
INFO:tensorflow:Step 11500 per-step time 0.711s loss=0.619
INFO:tensorflow:Step 11600 per-step time 0.698s loss=3.812
INFO:tensorflow:Step 11700 per-step time 0.698s loss=5.212
INFO:tensorflow:Step 11800 per-step time 0.703s loss=550625.562
INFO:tensorflow:Step 11900 per-step time 0.704s loss=3951414016.000
INFO:tensorflow:Step 12000 per-step time 0.696s loss=3848328704.000
INFO:tensorflow:Step 12100 per-step time 0.712s loss=3739422208.000
```

Reducing the training rate seems to help. This probably means there is a sharp cliff somewhere in gradient
which we're falling over.

Reducing the training rate from 0.8 to 0.2 seems to have mitigated this at the cost of much slower initial convergence.

This could be todo with small data counts for one of the classes.


### To the Cloud

Google Cloud let's us speed this process up by using a cloud instance with an expensive GPU attached (~ $10 per day)

![GPU](gpu.png)

Starting a Google Cloud instance with an Ubuntu 20.04 base image and an attached GPU,
we can apply all of the setup steps we worked out in [retraining/Dockerfile](retraining/Dockerfile).

Confirming we have a working GPU:

![CGoogle Cloud GPU](google-cloud-gpu.png)

We can create a Google Cloud machine image of the setup instance for a faster restart next time.

Uploading the check points from our in house training we can resume where we left off.

Comparing the per-step time with our local hardware the K80 looks slightly quicker.

```
I0525 09:57:42.750132 140193608288064 model_lib_v2.py:680] Step 10400 per-step time 0.734s loss=737993.750
```


### Exporting the model

After training we can export the model as a saved model.

[retraining/export.bash](retraining/export.bash)

This can then be loaded directly into TensorFlow Serving and is ready so detect our objects.

```
 curl http://localhost:8501/v1/models/squirrelnet
{
 "model_version_status": [
  {
   "version": "1",
   "state": "AVAILABLE",
   "status": {
    "error_code": "OK",
    "error_message": ""
   }
  }
 ]
}
```


### Putting it all together

We can now write a script which will listen for the motion messages, call the TensowFlow model for object detections
and send notifications.

[listener](listener)
