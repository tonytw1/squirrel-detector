
## Detecting motion and capturing images

Motion does a good job of detecting movement and bounding boxes.

Let's use a python script to catch this event and publish the image and event data.

`on_motion_detected.py`

This script needs capture the image file path and the bounding box from Motion 
and encode them into a message.

The Motion `on_picture_save` event is able to give us the image file path and bounding box.
We can hook these together with this configuration line:

`on_picture_save python /home/pi/on_motion_detected.py %f %w %h %K %L %i %J`

We want to open the binary image file and encode it for inclution in a message.
base64 encoding should be enough.

We'll use an MQTT topic to get these message out of the camera device.


## Classifying motion

We'll have a message containing a still image with a bounding box enclosing the area of motion.
We want to phrase this and may be crop to around the bounding box. 

We can them send the image to a classification API; look Google Vision for starters.

The results from the classification should be republished into another MQTT topic.

MQTT from python requires:

```
pip3 install paho-mqtt
```



