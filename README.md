## Detecting motion and capturing images

Motion does a good job of detecting movement and creating image files and bounding boxes.

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

We'll need to encode the image file for inclution in a message.
Base64 encoding should be enough.

We'll use MQTT as the message format. MQTT is really practical about message sizes limits.
We can publish the motion messages to a MQTT topic which other machines can subscribe to.

This happens in this script:
`on_motion_detected.py`


## Detecting objects

We have a message containing a still image with a bounding box enclosing an area of motion.
We want to phrase this and maybe crop to the bounding box.

We can then send the image to an object detection API.
The results from the classification can be republished into another MQTT topic.


## Object detection APIs

### Google Vision

Google Vision is probably the gold standard for object detection and it has a nice python API.
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


Returns a 70Mb block of JSON which we can start picking through

```
{'predictions': [{'detection_classes': [17.0, 44.0, 16.0, 9.0, 47.0, 64.0, 64.0, ...
```


raw_detection_boxes
detection_scores
raw_detection_scores
detection_anchor_indices
detection_multiclass_scores
detection_classes
num_detections
detection_boxes
detection_scores
detection_classes

both have length = num_detections
maxes:
0.692303538
88.0

Looking at the first 3 elements of these lists:
```
0.692303538
0.335249633
0.320963889
17.0
44.0
16.0
```

This looks like a 69% confidences for a detection of a `17`

The detection_class is from `mscoco_label_map.pbtxt`
There are only 80 unique ids in the file. 17 is `cat`

So, it looks like the COCO model doesn't know about squirrels!




MQTT from python requires:

```
pip3 install paho-mqtt
```



