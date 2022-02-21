#!/usr/bin/python3
import sys
import os
import subprocess
import json

# Capture calls from Motion's on_picture_save event and publishes them to a MQTT topic
# in JSON format with the image base64 encoded.
num_argv = len(sys.argv)
if num_argv >=4:
	# Image filepath and dimensions have been pass to us
	# Do we have a readable image file path?
	image_filepath = sys.argv[1]
	if os.access(image_filepath, os.R_OK):
		# Encode image to a base64 string
		result = subprocess.run(['base64', image_filepath], stdout=subprocess.PIPE)
		image = result.stdout.decode('utf-8')
		image_filename = os.path.split(image_filepath)[1]
		message = {
			'image_filename': image_filename,
			'image_width': sys.argv[2],
        		'image_height': sys.argv[3],
			'image': image
		}

		if (num_argv == 8):
			# A bounding box was also been past to us
			message['bounding_box_center_x'] = sys.argv[4]
			message['bounding_box_center_y'] = sys.argv[5]
			message['bounding_box_width'] = sys.argv[6]
			message['bounding_box_height'] = sys.argv[7]

		# Publish to mqtt topic in json format. Stream the message through stdin to avoid argument size limits
		mosquitto = subprocess.Popen(['mosquitto_pub', '-s', '-t', 'motion'], stdin=subprocess.PIPE)
		mosquitto.stdin.write(str.encode(json.dumps(message)))
		mosquitto.stdin.close()
