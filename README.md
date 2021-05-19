
Motion does a good job of detecting movement and bounding boxes.


Let's use a python script to catch this event and publish the image and event data.

`on_motion_detected.py`

This script needs capture the image file path and the bounding box from Motion 
and encode them into a message.

The Motion `on_picture_save` event is able to give us the image file path and bounding box.
We can hook these together with this configuration line:

`on_picture_save python /home/pi/on_motion_detected.py %f %w %h %K %L %i %J`


We want to open the binary image file and encoded it for inclution on a message.
base64 encoding should be enough.



