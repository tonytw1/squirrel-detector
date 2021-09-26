import os
import json

directory = r'/tmp'
for entry in os.scandir(directory):
    if (entry.path.endswith(".json")):
        with open (entry.path, "r") as f:
            data = json.load(f)
            regions = data['regions']
            for region in regions:
                points = region['points']
                topLeft = points[0]
                bottomRight = points[2]
                if (topLeft['y'] > bottomRight['y']):
                    print(entry.path)
                    print(topLeft , " ", bottomRight)
