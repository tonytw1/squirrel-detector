#!/usr/bin/python3
# Rolls up today's annotated images into a timelapse video

from datetime import date
import logging
import sys
import boto3
import os
import tempfile
import subprocess

logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s')

aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
s3_bucket = os.environ.get('S3_BUCKET')

# Determine the date
today = date.today()
datestamp = today.strftime("%Y%m%d")
logging.info("Rolling up for: " + datestamp)

# Query bucket for matching images
s3_client = boto3.client('s3',
                         aws_access_key_id=aws_access_key_id,
                         aws_secret_access_key=aws_secret_access_key)

paginator = s3_client.get_paginator('list_objects')
page_iterator = paginator.paginate(
    Bucket= s3_bucket,
    Prefix='annotated/' + datestamp,
    PaginationConfig={
        'MaxItems': 10000,
        'PageSize': 100
    }
)

files = []
for page in page_iterator:
    if 'Contents' in page:
        page_contents = page['Contents']
        for file in page_contents:
            files.append(file['Key'])

logging.info("Found " + str(len(files)) + " files to rollup")
# Bailout if nothing todo
if len(files) == 0:
    sys.exit()

# Download matching images to a working directory

tmp = tempfile.mkdtemp(suffix=None, prefix='motionrollup', dir=None)
for file in files:
    localpath = tmp + '/' + os.path.split(file)[1]
    logging.info("Downloading  " + file + " to " + localpath)
    s3_client.download_file(s3_bucket, file, localpath)

# Invoke ffmpeg
input_glob = "/" + str(tmp) + "/*.jpg"
output_filename = datestamp + ".mp4"
output_path = tmp + "/" + datestamp + ".mp4"
logging.info("ffmpeg rendering to: " + output_path)
subprocess.call(["ffmpeg", "-framerate", "10", "-pattern_type", "glob", "-i", input_glob, output_path])

# Update output to bucket
upload_path = "rollups/" + output_filename
logging.info("Uploading to s3: " + upload_path)
s3_client.upload_file(output_path, s3_bucket, upload_path)

logging.info("Done")
sys.exit
