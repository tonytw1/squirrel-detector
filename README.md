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

Saved models have been trained and can be downloaded and used to run predictions against our images.
So lets pick a model from the Model Zoo and try to run it against one of our test images.

Working on a local machine I was blocked almost immediately with an error trying to use the loaded model.
This could be a mismatch between TensorFlow 2.5 and the available examples.models

Retreating to Google Colab notebooks offers a known good development environment.
The same problem persisted in Colab. Reverting from a TensorFlow2 model to a TensorFlow1 model resolved it.

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

Looking up class 17 in the label file we find `cat`.

It looks like the model doesn't know about squirrels!
Squirrels are not one of the classes this model was trained on.

Back porting what we learnt in the Colab worksheet we can create a local script which can make the same prediction.
This is `detect.py`. There is plentt in there which I don't yet understand.

## Running a model with TensorFlow Serving

"TensorFlow Serving makes it easy to deploy new algorithms and experiments, while keeping the same server architecture and APIs"

So lets use this to setup a local object detection service.

https://www.tensorflow.org/tfx/serving/docker


Checking for available models
http://localhost:8501/v1/models/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8

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


