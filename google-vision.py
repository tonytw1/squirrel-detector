import io
import os

# Imports the Google Cloud client library
from google.cloud import vision

# Instantiates a client
client = vision.ImageAnnotatorClient()

# The name of the image file to annotate
file_name = os.path.abspath('images/squirrel2.jpg')

# Loads the image into memory
with io.open(file_name, 'rb') as image_file:
    content = image_file.read()

image = vision.Image(content=content)

objects = client.object_localization(image=image).localized_object_annotations

print('Number of objects found: {}'.format(len(objects)))
for object_ in objects:
	print('\n{} (confidence: {})'.format(object_.name, object_.score))
	print('Normalized bounding polygon vertices: ')
	for vertex in object_.bounding_poly.normalized_vertices:
		print(' - ({}, {})'.format(vertex.x, vertex.y))

