# from jigsaw import train_generator
from tqdm import tqdm
import tensorflow as tf
from pathlib import Path
from gensim.models import FastText
import pandas as pd


BATCH_SIZE = 1


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _parse_function(proto):

    key_to_feature = {
        'image_width': tf.io.FixedLenFeature([], tf.int64),
        'image_height': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }

    # load one example
    parsed_features = tf.io.parse_single_example(proto, key_to_feature)

    # turn your saved image string into an array
    parsed_features['image'] = tf.io.decode_raw(parsed_features['image'], tf.uint8)

    return parsed_features['image'], parsed_features['label']


def create_dataset(filepath=Path('/mnt/sd1/DS/tfrekordy')):
    dataset = tf.compat.v1.data.TFRecordDataset([str(item) for item in filepath.glob('*')])

    # maps the parser on every filepath in the array. set the number of parallel loaders here
    dataset = dataset.map(_parse_function, num_parallel_calls=8)

    # the dataset will go on forever
    dataset = dataset.cache().repeat()

    # set the number of datasets you want to load and shuffle
    dataset = dataset.shuffle(BATCH_SIZE)

    # set the batchsiye
    dataset = dataset.batch(BATCH_SIZE)

    # create an iterator
    # iterator = dataset.make_one_shot_iterator()
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)

    # create your tf representation of the iterator
    # image, label = iterator.get_next()
    image, label, image_height, image_width = iterator.get_next()

    # bring a picture back in shape
    image = tf.reshape(image, [-1, image_width, image_height, 1])
    # one hot array
    NUM_CLASSES = 2
    label = tf.one_hot(label, NUM_CLASSES)
    # label = tf.reshape(label, [-1, NUM_CLASSES])

    return image, label


def write_to_tfrecords(writer, batch):

    image = batch['image'].tobytes()
    example_proto = tf.train.Example(features=tf.train.Features(feature={
        'image_width': _int64_feature(batch['image'].shape[0]),
        'image_height': _int64_feature(batch['image'].shape[1]),
        'image': _bytes_feature(image),
        'label': _int64_feature(batch['label'])
    }))

    writer.write(example_proto.SerializeToString())


if __name__ == '__main__':
    path = '/mnt/sd1/DS/jigsaw/jigsaw-toxic-comment-train-processed-seqlen128.csv'
    table = pd.read_csv(path)
    model = FastText.load("/mnt/sd1/DS/word2vec_fast_text.model")

    batch_size = 10000
    batch = {
        "image": [],
        "label": []
    }
    result_tf_file = '/mnt/sd1/DS/tfrekordy/tf_file_1'
    writer = tf.io.TFRecordWriter(result_tf_file)

    batch_index = 0
    counter = 2
    for X, y in train_generator(table, model):
        batch["image"] = X
        batch["label"] = y
        write_to_tfrecords(writer, batch)
        batch_index += 1
        if batch_index == batch_size:
            writer = tf.io.TFRecordWriter('/mnt/sd1/DS/tfrekordy/tf_file_{}'.format(counter))

            batch_index = 0
            counter += 1







