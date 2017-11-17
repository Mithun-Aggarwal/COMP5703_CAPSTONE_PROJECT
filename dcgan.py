# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 22:35:32 2017

@author: magg5201
For COMP 5703 - Capstone project 
Unsupervised learning with Gan for LF data

The main model file that defines the network and its components
"""

#imported libraries
from config import config
import tensorflow as tf
import numpy as np
import ops
from tfrecords_reader import TFRecordsReader

FLAGS = tf.app.flags.FLAGS

#The final inference model 
def inference(images, z):
    generated_images = generator(z)
    
    D_logits_real = discriminator(images)

    D_logits_fake = discriminator(generated_images, reuse=True)

    return D_logits_real, D_logits_fake, generated_images

#The fuction defining all the losses
def loss(D_logits_real, D_logits_fake):
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_real,
                                                labels=tf.ones_like(D_logits_real)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_fake,
                                                labels=tf.zeros_like(D_logits_fake)))
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_fake,
                                                labels=tf.ones_like(D_logits_fake)))
    d_loss = d_loss_real + d_loss_fake

    return d_loss, g_loss, d_loss_real , d_loss_fake

#the function defining the training
def train(d_loss, g_loss):
    # variables for discriminator
    d_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    # variables for generator
    g_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

    # train discriminator
    d_optimzer = tf.train.AdamOptimizer(FLAGS.learning_rate_dis, beta1=FLAGS.beta1)
    train_d_op = d_optimzer.minimize(d_loss, var_list=d_vars)

    # train generator
    g_optimzer = tf.train.AdamOptimizer(FLAGS.learning_rate_gen, beta1=FLAGS.beta1)
    train_g_op = g_optimzer.minimize(g_loss, var_list=g_vars)
    
    return train_d_op, train_g_op

#The function defining the discriminator
def discriminator(images, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()

        # conv1
        conv1 = ops.conv_2d(images, 64, scope="conv1")

        # leakly ReLu
        h1 = ops.leaky_relu(conv1)
        
        # conv2
        conv2 = ops.conv_2d(h1, 128, scope="conv2")

        # batch norm
        norm2 = ops.batch_norm(conv2, scope="batch_norm2", is_training=True)
        #print(type(norm2))
        
        # leaky ReLU
        h2 = ops.leaky_relu(norm2)
        #print(type(h2))
        
        # conv3
        conv3 = ops.conv_2d(h2, 256, scope="conv3")
        #print(type(conv3))
        
        # batch norm
        norm3 = ops.batch_norm(conv3, scope="batch_norm3", is_training=True)
        #print(type(norm3))
        
        # leaky ReLU
        h3 = ops.leaky_relu(norm3)
        #print(type(h3))
        
        # conv4
        conv4 = ops.conv_2d(h3, 512, scope="conv4")
        #print(type(conv4))
        
        # batch norm
        norm4 = ops.batch_norm(conv4, scope="batch_norm4", is_training=True)

        # leaky ReLU
        h4 = ops.leaky_relu(norm4)

        # reshape
        h4_reshape = tf.reshape(h4, [FLAGS.batch_size, -1])

        # fully connected 
        fc = ops.fc(h4_reshape, 1, scope="fc")

        return fc

#The function defining the genrator
def generator(z):
    with tf.variable_scope("generator") as scope:

        # project z and reshape
        oh, ow = FLAGS.output_image_height, FLAGS.output_image_width
        z_ = ops.fc(z, 512 * int(oh / 16) * int(ow / 16), scope="project")
        z_ = tf.reshape(z_, [-1, int(oh / 16), int(ow / 16), 512])

        # batch norm
        norm0 = ops.batch_norm(z_, scope="batch_norm0", is_training=True)

        # ReLU
        h0 = tf.nn.relu(norm0)

        # conv1
        conv1 = ops.conv2d_transpose(
            h0, [FLAGS.batch_size, int(oh / 8), int(ow / 8), 256],
            scope="conv_tranpose1")

        # batch norm
        norm1 = ops.batch_norm(conv1, scope="batch_norm1", is_training=True)

        # ReLU
        h1 = tf.nn.relu(norm1)

        # conv2
        conv2 = ops.conv2d_transpose(
            h1, [FLAGS.batch_size, int(oh / 4), int(ow / 4), 128],
            scope="conv_tranpose2")

        # batch norm
        norm2 = ops.batch_norm(conv2, scope="batch_norm2", is_training=True)

        # ReLU
        h2 = tf.nn.relu(norm2)

        # conv3
        conv3 = ops.conv2d_transpose(
            h2, [FLAGS.batch_size, int(oh / 2), int(ow / 2), 64], scope="conv_tranpose3")

        # batch norm
        norm3 = ops.batch_norm(conv3, scope="batch_norm3", is_training=True)

        # ReLU
        h3 = tf.nn.relu(norm3)

        # conv4
        conv4 = ops.conv2d_transpose(
            h3, [FLAGS.batch_size, oh, ow, FLAGS.output_image_channel],
            scope="conv_tranpose4")

        # tanh
        h4 = tf.nn.tanh(conv4)

    return h4

#the function intracting with TFRecord later and provinf batched output according to the model
def inputs(batch_size=FLAGS.batch_size):
    crop_height, crop_width = FLAGS.input_image_height, FLAGS.input_image_width

    if FLAGS.dataset == "LF":
        reader = TFRecordsReader(
            image_height=FLAGS.input_image_height,
            image_width=FLAGS.input_image_width,
            image_channels=FLAGS.input_image_channel,
            image_format=FLAGS.import_image_format,
            directory=FLAGS.output_directory,
            filename_pattern= FLAGS.import_filename_pattern,
            crop=FLAGS.crop,
            crop_height=crop_height,
            crop_width=crop_width,
            resize=FLAGS.resize,
            resize_height=FLAGS.output_image_height,
            resize_width=FLAGS.output_image_width,
            num_examples_per_epoch=FLAGS.ephoo_size)

        images, label = reader.inputs(batch_size=FLAGS.batch_size)
        float_images = tf.cast(images, tf.float32)
        float_images = float_images / 127.5 - 1.0
        #print(tf.print(label))


    return float_images,label

#The Function to add the filters on tensorboard 
def addfilterstensorboard(filters=5):
    with tf.variable_scope('discriminator/conv1', reuse=True):
        Kernal = tf.get_variable("w")
        grid = ops.put_kernels_on_grid (Kernal,1)
        with tf.name_scope("Conv_filter_1"):
            tf.summary.image(name ='Conv_filter_1',tensor =grid,max_outputs=1, collections=["Generated_filters"])
    with tf.variable_scope('generator/conv_tranpose4', reuse=True):
        Kernal2 = tf.get_variable("w")
        grid2 = ops.put_kernels_on_grid (Kernal2,1)
        tf.summary.image(name ='conv_tranpose4',tensor =grid2,max_outputs=1, collections=["Generated_filters"])

