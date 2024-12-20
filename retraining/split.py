import tensorflow as tf
import os

files = tf.data.Dataset.list_files("data/*.tfrecord")
ds = tf.data.TFRecordDataset(files)
ds = ds.shuffle( buffer_size=10000)

l = list(ds.as_numpy_iterator())

# The tfrecond tensor format is really basic; we need to know about the fields in the schema
LABELED_TFREC_FORMAT = {
    "image/filename": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
    "image/object/class/text": tf.io.VarLenFeature(tf.string),  # shape [] means single element
}

for x in l:
    # Show that we can extract filename and tagged classes from each record
    record = tf.io.parse_single_example(x, LABELED_TFREC_FORMAT)
    filenameTensor = record['image/filename']
    filename = bytes.decode(filenameTensor.numpy())

    classTensor = record['image/object/class/text']
    values = classTensor.values
    classNames = []
    for v in values:
        className = bytes.decode(v.numpy())
        classNames.append(className)
    print(filename, classNames)


total = len(l)
print("Loaded records in dataset:", total)

evalSize = round(total * 0.20)
trainingSize = total - evalSize
print("Eval size", evalSize)
print("Training size", trainingSize)

# Can't use ds.take and ds.skip because the order is not repeatable; duplicate items
eval = l[0:evalSize]
training = l[evalSize:total]

print("Extracted ", len(eval), " eval items")
print("Left with ", len(training), " training items")

print("Moving eval files")
for x in eval:
   record = tf.io.parse_single_example(x, LABELED_TFREC_FORMAT)
   filenameTensor = record['image/filename']
   filename = bytes.decode(filenameTensor.numpy())
   tfrecordFilename = filename.replace('.jpg', '.tfrecord')
   os.rename('data/' + tfrecordFilename, 'eval/' + tfrecordFilename)


print("Moving training files")
for x in training:
   record = tf.io.parse_single_example(x, LABELED_TFREC_FORMAT)
   filenameTensor = record['image/filename']
   filename = bytes.decode(filenameTensor.numpy())
   tfrecordFilename = filename.replace('.jpg', '.tfrecord')
   os.rename('data/' + tfrecordFilename, 'training/' + tfrecordFilename)
