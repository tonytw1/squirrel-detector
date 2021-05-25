## Hardware

We're using a [Raspberry Pi Zero W](https://www.raspberrypi.org/products/raspberry-pi-zero-w/) with the
[Camera Module V2](https://www.raspberrypi.org/products/camera-module-v2/).

This gives us Wifi, 1 core and 512Mb of memory.

The camera module appears as a Video4Linux device.
You can see device details with this command:
```
v4l2-ctl --all
```


## Detecting motion and capturing images
[Motion](https://motion-project.github.io) is available as a Raspberry Pi package.
It does a good job of detecting movement and creating image files and bounding boxes.

Here's an example of Motion detecting and bounding a movement:
![This is not a squirrel](images/not_squirrel.jpg)


We'd like Motion to detect bounding boxes but not draw them on the saved image files.
This configuration line seems todo this:
```
locate_motion_mode preview
```

Let's use a python script to catch these events and publish them.

This script needs to capture the image file path and the bounding box from Motion and encode them into a message.

The Motion `on_picture_save` event is able to give us the image file path and bounding box.
We can hook these together with this configuration line:

`on_picture_save python3 /home/pi/on_motion_detected.py %f %w %h %K %L %i %J`

Our camera is connected to a small device with limited processing capability.
We want to send the image somewhere where a more capable machine can look at it.

We'll need to encode the image file for inclusion in a message.
Base64 encoding should be enough.

We'll use MQTT as the message format. MQTT is really practical about message sizes limits.
We can publish the motion messages to a MQTT topic which other machines can subscribe to.

This happens in this script:
`on_motion_detected.py`


## Detecting objects

We now have a message containing a still image with a bounding box enclosing an area of motion.
We want to phrase this and maybe crop to the bounding box.

We can then send the image to an object detection API.
The results from the classification can be republished into another MQTT topic.


## Object detection APIs

How are we going to classify the moving objects Motion has detected?
We'll like to pass the area of interest to an object detection API.


### Google Vision

Google Vision is probably the gold standard for object detection and has a nice python API.
It seems to know all about squirrels as well.
Here's a script to detect objects an image file and it's sample output.

![Google Vision output](google_vision.png)


### Local alternatives?

The free tier for Google Vision isn't suited for continuous use.
Is there anything we can run locally?

Pretrained TensorFlow object detection models are available and running one locally might be an interesting side quest.

There are 2 interesting problems here. Can we find a model which can detect the objects we're interested in and
can we easily run it and call it locally.

The model will need to be wrapped in some sort of API so we can call it from our message handling script.


## TensorFlow object detection models

Saved models have been pretrained and can be downloaded and used to run predictions against our images.

Lets pick a model from the [TensorFlow Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) and try to run it against one of our test images.

Working on a local machine I was blocked almost immediately with an error trying to use the loaded model.
This could be a mismatch between TensorFlow 2.5 and the available examples.

Retreating to Google Colab notebooks offers a known good development environment.

Alot of data development work goes in in notebook environments like Jupyter. The data community have discovered a really
interesting way of working here. I'd encourage an software developer who haven't seen this before to have a look.


### Testing a saved model in Colab

With a saved model imported into our Colab notebook we can load one of test images and ask the model to predict the visible objects.

![Colan perdiction](colab.png)

Requesting a prediction.

![Model predict](predict.png)

The prediction returns a large map of results.

![Prediction results](predictions.png)


`detection_classes` and `detection_scores` are interesting.
This turns out to mean a 73% confidence of a class 17 object.

What are the classes and why are the values all below 100?
The saved model was trained on the COCO image set.
It was only traught about 80 unique objects and the classe id's refer to one of them.

The detection_class labels are available in the file `mscoco_label_map.pbtxt`

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

Looking up class 17 in the  label file we find `cat`.

It looks like the model doesn't know about squirrels!
Squirrels are not one of the classes this model was trained on.

### Local detection script

Back porting what we learnt in the Colab worksheet we can create a local script which can make the same prediction.
This is `detect.py`. There is plenty in there which I don't yet understand.

We've now verified that we can use TensorFlow to run a pretrained model locally.
That pretrained model doesn't know about the specific animals we're interested in but can probably be retrained.

Let's move onto productionising what we have on the assumption we'll be able to improve the model with training.



## Running a model with TensorFlow Serving

"[https://www.tensorflow.org/tfx/serving/docker](TensorFlow Serving) makes it easy to deploy new algorithms and experiments, while keeping the same server architecture and APIs"

This looks exactly what we want. A way to deploy a saved model behind a REST API.

A Docker container is provided. 
We use this as a base image to produce an image with our model baked into it.
`Dockerfile`


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




Now we can ask for a predicition with an HTTP call rather than importing the TensorFlow model into the script

`
detect_rest.py
`



### Hooking it all together

We can now write a script which will listen for the motion messages and call the TensowFlow model for object detections.

`listener.py'







## Retraining

### Collecting training data

Unlike humans animals won't generally give out personally identifying informational for free.
They will trade it for nuts though.


Collecting several day's images gave a collection of several hundred training images with examples of most of the garden animals.



### Annotating images

[VoTT (Visual Object Tagging Tool)](https://github.com/microsoft/VoTT) 

230 in 30 minutes. This was much quicker than expected and a somewhat cathartic.

VoTT can export direct to the format TensorFlow needs.


### Split the data




### Object Detection API

Don't try todo this locally; use a Docker image for GPU support
https://www.tensorflow.org/install/docker



https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md

retraining/Dockerfile



gtx 1050 Ti 4Gb
```
I0524 17:03:34.777422 139797117065024 model_lib_v2.py:680] Step 400 per-step time 1.050s loss=1.978
```


### Checkpoints

As it trains TensorFlow periodically drops check points.
These represent the current parameter settings for the model.
Training is about finding the model parameters which fit the data.

![Check points](checkpoints.png)

Check points can be used to port pause and resume training.
They can also be used to resume training on a faster GPU enabled cloud instance.


### To the Cloud

Starting a Google Cloud instance with an Ubuntu 20.04 base image and an attached GPU,
we can apply all of the setup
steps we worked out in the `retrain/Dockerfile`


![CGoogle Cloud GPU](google-cloud-gpu.png)


Google Cloud 
Tesla K80



We can create a machine image of the setup instance for a faster restart next time.



