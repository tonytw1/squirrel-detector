#!/usr/bin/python3
import sys
import os
import subprocess
import json

# Do we have a readable image file path?
image_filepath = sys.argv[1];


if os.access(image_filepath, os.R_OK):
        # Encode it to a base64 string
        result = subprocess.run(['base64', image_filepath], stdout=subprocess.PIPE)
        image = result.stdout.decode('utf-8')
        message = ""
        for arg in sys.argv[2:]:
                message += arg + ":"

        message = {
                'image_file': image_filepath,
                'image_width': sys.argv[2],
                'image_height': sys.argv[3],
                'bounding_box_center_x': sys.argv[4],
                'bounding_box_center_y': sys.argv[5],
                'bounding_box_width': sys.argv[6],
                'bounding_box_height': sys.argv[7],
                'image': image
        }

        subprocess.run(['mosquitto_pub', '-t', 'motion', '-m', json.dumps(message)])

