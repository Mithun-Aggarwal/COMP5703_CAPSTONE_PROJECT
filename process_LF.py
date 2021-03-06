# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 22:35:32 2017

@author: magg5201
For COMP 5703 - Capstone project 
Unsupervised learning with Gan for LF data

Converts LF images to TFRecords file format with Example protos.
"""
#Import Libaries
from datetime import datetime
import os
import sys
import random
import threading
from config import config
import tensorflow as tf
import numpy as np
import re

FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename,label, image_buffer, height, width):
    """Build an Example proto for an example.
  """
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'channels': _int64_feature(FLAGS.import_channels),
        'format': _bytes_feature(tf.compat.as_bytes(FLAGS.import_image_format)),
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(tf.compat.as_bytes(image_buffer))
    }))
    return example


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        self._sess = tf.Session()
        if (FLAGS.import_image_format == "png"):
            self._decode_png_data = tf.placeholder(dtype=tf.string)
            self._decode_png = tf.image.decode_png(
                self._decode_png_data, channels=FLAGS.import_channels)
        elif (FLAGS.import_image_format == "jpg"):
            self._decode_jpec_data = tf.placeholder(dtype=tf.string)
            self._decode_jpec = tf.image.decode_jpeg(
                self._decode_jpec_data, channels=FLAGS.import_channels)
        else:
            print(" ImageCoder Wrong input type")
            
    def decode_jpeg(self, image_data):
        image = self._sess.run(
            self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == FLAGS.import_channels
        assert image.shape[2] == FLAGS.import_channels
        return image

    def decode_png(self, image_data):
        image = self._sess.run(
            self._decode_png, feed_dict={self._decode_png_data: image_data})
        assert len(image.shape) == FLAGS.import_channels
        assert image.shape[2] == FLAGS.import_channels
        return image      
    
def _process_image(filename, coder):
    """Process a single image file.
  """
    # Read the image file.
    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()

    regex = re.compile(r'\d+')
    img_sub_num =  int(regex.findall(filename)[1])
    # Decode the RGB JPEG.
    if (FLAGS.import_image_format == "png"):
        image = coder.decode_png(image_data)
    elif (FLAGS.import_image_format == "jpg"):
        image = coder.decode_jpeg(image_data)
    else:
        print("_process_image Wrong input")
    # Check that image converted to RGB
    assert len(image.shape) == FLAGS.import_channels
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == FLAGS.import_channels
    

    return image_data,img_sub_num, height, width
    
def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
                               num_shards):
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)

    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]
    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_directory, output_filename)

        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(
            shard_ranges[s], shard_ranges[s + 1], dtype=int)

        for i in files_in_shard:
            filename = filenames[i]

            image_buffer,label, height, width = _process_image(filename, coder)

            example = _convert_to_example(filename,label, image_buffer, height,
                                          width)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 1000:
                print(
                    '%s [thread %d]: Processed %d of %d images in thread batch.'
                    % (datetime.now(), thread_index, counter,
                       num_files_in_thread))
                sys.stdout.flush()

        writer.close()
        print('%s [thread %d]: Wrote %d images to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0

    print('%s [thread %d]: Wrote %d images to %d shards.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()
    
def _process_image_files(name, filenames, num_shards):
    """Process and save list of images as TFRecord of Example protos.
    """

    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(filenames),
                          FLAGS.num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' %
          (FLAGS.num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = ImageCoder()
    
    threads = []
    for thread_index in range(len(ranges)):
        args = (coder, thread_index, ranges, name, sorted(filenames), num_shards)
        #sorted(filenames)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
          (datetime.now(), len(filenames)))
    sys.stdout.flush()
    

def _find_image_files(data_dir):
    
    if (FLAGS.import_image_format == "png"):
        png_file_path = '%s/*.png' % data_dir
        filenames = tf.gfile.Glob(png_file_path)
    elif (FLAGS.import_image_format == "jpg"):
        jpeg_file_path = '%s/*.jpg' % data_dir
        filenames = tf.gfile.Glob(jpeg_file_path)
    else:
        print("Check the input type")
    # shuffle files
    if(FLAGS.file_name_shuffling):
        shuffled_index = list(range(len(filenames)))
        random.seed(12345)
        random.shuffle(shuffled_index)
        filenames = [filenames[i] for i in shuffled_index]

    return filenames

def _process_dataset(name, directory, num_shards):
    """Process a complete data set and save it as a TFRecord.
    """
    
    filenames = _find_image_files(directory)
    #print(filenames)
    _process_image_files(name, filenames, num_shards)
