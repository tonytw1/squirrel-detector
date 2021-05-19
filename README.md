
Motion does a good job of detecting movement and bounding boxes.


Let's use a pyhton script to response to this event and publish the image and data about the event.
`on_motion_detected.py`
This script want to capture the image file path and the bounding box and
try to encode both of these into a message.

The `on_picture_save` event seems to be able to give us the image file and bounding box.
We can hook these together with this configuration line:

`on_picture_save python /home/pi/on_motion_detected.py %f %w %h %K %L %i %J`

