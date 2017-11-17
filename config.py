# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 22:35:32 2017

@author: magg5201
For COMP 5703 - Capstone project 
Unsupervised learning with Gan for LF data

This is the main Config file.
Flags are also prefilled in this file only.
"""

#imported libraries
from easydict import EasyDict as edict
import tensorflow as tf


#All the edicts pipeline
config = edict()
config.PRE_TRAIN = edict()
config.TRAIN = edict()
config.TEST = edict()
config.DCGAN = edict()
config.OPS = edict()
config.ORIGNAL = edict()
config.GENERATOR = edict()
config.SETTINGS = edict()

## PRE_TRAINING
#Examples in each lf_file
config.PRE_TRAIN.shards = 1
config.PRE_TRAIN.threads = 1
config.PRE_TRAIN.file_name_shuffling = 'False'
config.PRE_TRAIN.import_colorspace = 'RGB'
config.PRE_TRAIN.import_channels = 3
config.PRE_TRAIN.import_image_format = "png"
config.PRE_TRAIN.import_filename_pattern = 'LF-*'

#input LF image size
# Data Config 375, 541
config.ORIGNAL.BIG_IMG_H_PATCH = 5250
config.ORIGNAL.BIG_IMG_W_PATCH = 7574
config.ORIGNAL.BIG_IMG_C_PATCH = 3
# Window options
config.ORIGNAL.CUT_DIM_X = 14
config.ORIGNAL.CUT_DIM_Y = 14
#Window image extraction
config.ORIGNAL.CUT_X = 4
config.ORIGNAL.CUT_Y = 2

#The main configurtion responsible for controling the flow
config.SETTINGS.agumentation_rotate = "False"
config.SETTINGS.imgprocessing_status = "True"
config.SETTINGS.TF_LF_creation = "True"
config.SETTINGS.Train_status = "True"
config.SETTINGS.restore_model = "False"

#The orignal size of the subview
#DCGAN-INPORT
config.DCGAN.input_image_height = 375
config.DCGAN.input_image_width = 541
config.DCGAN.input_image_channel = 3

#the size of images feeded and extracted from the network
config.DCGAN.output_image_height = 32
config.DCGAN.output_image_width = 32
config.DCGAN.output_image_channel = 3

#DGGAN-Generator
config.GENERATOR.D_H_PATCH = 10  
config.GENERATOR.D_W_PATCH = 10
config.GENERATOR.D_PATCH = 100
config.GENERATOR.D_C_PATCH = 1

#OPS-
config.OPS.conv_2d_kernal = 5
config.OPS.conv_2d_strides = 2
config.OPS.conv_2d_stddev = 0.02
tf.app.flags.DEFINE_integer('conv_2d_kernal', config.OPS.conv_2d_kernal, 'conv_2d_kernal')
tf.app.flags.DEFINE_integer('conv_2d_strides', config.OPS.conv_2d_strides, 'conv_2d_strides')
tf.app.flags.DEFINE_float('conv_2d_stddev', config.OPS.conv_2d_stddev, 'conv_2d_stddev')

#DCGAN-SETTINGS
config.SETTINGS.filters = 16
config.SETTINGS.gpu_use = 0.25000
config.SETTINGS.crop = 'False'
config.SETTINGS.resize = 'True'

## Training (LF)
config.TRAIN.steps = 450000
config.TRAIN.batch_size = 4
config.TRAIN.ephoo_size = 5
config.TRAIN.extra_Training_discriminator = 0
config.TRAIN.summary_save_step = 10
config.TRAIN.model_save_step = 10

#config.TRAIN.lr_decay = 0.05
config.TRAIN.Samples_number = 4
config.TRAIN.Samples_Grid_X = 2
config.TRAIN.Samples_Grid_Y = 2
config.TRAIN.lr_gen = 0.002
config.TRAIN.lr_dis = 0.0002
config.TRAIN.beta1 = 0.5

#Directories
#windows
#==============================================================================
# config.PRE_TRAIN.big_img_directory = 'C:/Users/ksxl806/Documents/PythonScripts/LF_DCGAN/LF_DCGAN/BIG_IMG'
# config.PRE_TRAIN.directory = "C:/Users/ksxl806/Documents/PythonScripts/LF_DCGAN/LF_DCGAN/RAW_IMAGE_375_571/"
# config.PRE_TRAIN.directory_read = "C:/Users/ksxl806/Documents/PythonScripts/LF_DCGAN/LF_DCGAN/RAW_IMAGE_375_571"
# config.PRE_TRAIN.output_directory = 'C:/Users/ksxl806/Documents/PythonScripts/LF_DCGAN/LF_DCGAN/TF_IMAGES_375_571'
# config.LOG_LOCATION = "C:/Users/ksxl806/Documents/PythonScripts/LF_DCGAN/LF_DCGAN/TLogs/TLogs_LF_DCGAN_377_547"
# config.CHECKPOINT_LOCATION = "C:/Users/ksxl806/Documents/PythonScripts/LF_DCGAN/LF_DCGAN/checkpoints/LF_DCGAN_377_547"
# config.TEST.directory = "C:/Users/ksxl806/Documents/PythonScripts/LF_DCGAN/LF_DCGAN/RAW_IMAGE_TEST_375_571/"
#==============================================================================

#==============================================================================
# #server
# config.PRE_TRAIN.big_img_directory = '/media/data/dataset/Cars1'
# config.PRE_TRAIN.directory = '/media/data/dataset/RAW_IMAGE_375_541/'
# config.PRE_TRAIN.directory_read = '/media/data/dataset/RAW_IMAGE_375_541'
# config.PRE_TRAIN.output_directory = '/media/data/dataset/TF_IMAGES_375_541'
# # Save Directories
# config.LOG_LOCATION = "/home/mithun/Log/TLogs/TLogs_F_375_541_B_64"
# config.CHECKPOINT_LOCATION = "/home/mithun/Log/checkpoints/LF_DCGAN_F_375_541_B_64"
# config.TEST.directory = '/media/data/dataset/RAW_TEST_IMAGE_375_541'
#==============================================================================

#mac
config.PRE_TRAIN.big_img_directory = '/Users/MIT/Downloads/BIG_IMG'
config.PRE_TRAIN.directory = '/Users/MIT/Downloads/LF_DCGAN/RAW_IMAGE_INA_1_W_14_14_4_2/'
config.PRE_TRAIN.directory_read = "/Users/MIT/Downloads/LF_DCGAN/RAW_IMAGE_INA_1_W_14_14_4_2"
config.PRE_TRAIN.output_directory = '/Users/MIT/Downloads/LF_DCGAN/TF_IMAGES_INA_64_S_1_T_1'
# Save Directories
config.LOG_LOCATION = "/Users/MIT/Downloads/LF_DCGAN/Log/Tlog/TLOG_I_1_S_32_32_L_10_B_4"
config.CHECKPOINT_LOCATION = "/Users/MIT/Downloads/LF_DCGAN/Log/Checkpoint/LF_DCGAN_I_1_S_32_32_L_10_B_4"
config.TEST.directory = '/Users/MIT/Downloads/LF_DCGAN/RAW_IMAGE_TEST_INA_1_W_14_14_4_2/'

#All Flags
tf.app.flags.DEFINE_string('dataset', 'LF', 'name of data set')
tf.app.flags.DEFINE_integer('BIG_IMG_H_PATCH', config.ORIGNAL.BIG_IMG_H_PATCH, 'input image height')
tf.app.flags.DEFINE_integer('BIG_IMG_W_PATCH', config.ORIGNAL.BIG_IMG_W_PATCH, 'input image width')
tf.app.flags.DEFINE_integer('BIG_IMG_C_PATCH', config.ORIGNAL.BIG_IMG_C_PATCH, 'input image channels')
tf.app.flags.DEFINE_integer('CUT_DIM_X', config.ORIGNAL.CUT_DIM_X, 'CUT_DIM_X')
tf.app.flags.DEFINE_integer('CUT_DIM_Y', config.ORIGNAL.CUT_DIM_Y, 'CUT_DIM_Y')
tf.app.flags.DEFINE_integer('CUT_X', config.ORIGNAL.CUT_X, 'CUT_X')
tf.app.flags.DEFINE_integer('CUT_Y', config.ORIGNAL.CUT_Y, 'CUT_Y')
tf.app.flags.DEFINE_string('test_path', config.TEST.directory, 'test_path')
tf.app.flags.DEFINE_integer('input_image_height', config.DCGAN.input_image_height, 'input image height')
tf.app.flags.DEFINE_integer('input_image_width', config.DCGAN.input_image_width, 'input image width')
tf.app.flags.DEFINE_integer('input_image_channel', config.DCGAN.input_image_channel, 'input image channel')
tf.app.flags.DEFINE_string('import_image_format', config.PRE_TRAIN.import_image_format, 'import image format')
tf.app.flags.DEFINE_string('import_filename_pattern', config.PRE_TRAIN.import_filename_pattern, 'input image format')
tf.app.flags.DEFINE_integer('output_image_height', config.DCGAN.output_image_height, 'output image height')
tf.app.flags.DEFINE_integer('output_image_width', config.DCGAN.output_image_width, 'output image width')
tf.app.flags.DEFINE_integer('output_image_channel', config.DCGAN.output_image_channel, 'output image channel')
tf.app.flags.DEFINE_integer('z_dim', config.GENERATOR.D_PATCH, 'generator input dim')
tf.app.flags.DEFINE_integer('num_filters', config.SETTINGS.filters, 'number of filters')
tf.app.flags.DEFINE_boolean('crop', config.SETTINGS.crop, 'crop image or not')
tf.app.flags.DEFINE_boolean('resize', config.SETTINGS.resize, 'resize image or not')
tf.app.flags.DEFINE_float('gpu_use', config.SETTINGS.gpu_use, 'gpu_use')
tf.app.flags.DEFINE_string('restore_model', config.SETTINGS.restore_model, 'restore_model')
tf.app.flags.DEFINE_string('imgprocessing_status', config.SETTINGS.imgprocessing_status, 'imgprocessing_status')
tf.app.flags.DEFINE_string('Train_status', config.SETTINGS.Train_status, 'Train_status')
tf.app.flags.DEFINE_string('TF_LF_creation', config.SETTINGS.TF_LF_creation, 'TF_LF_creation')
tf.app.flags.DEFINE_string('agumentation_rotate', config.SETTINGS.agumentation_rotate, 'agumentation_rotate')
tf.app.flags.DEFINE_integer('batch_size', config.TRAIN.batch_size, 'batch size')
tf.app.flags.DEFINE_integer('ephoo_size', config.TRAIN.ephoo_size, 'ephoo size')
tf.app.flags.DEFINE_integer('extra_Training_discriminator', config.TRAIN.extra_Training_discriminator, 'extra Training discriminator')
tf.app.flags.DEFINE_integer('summary_save_step', config.TRAIN.summary_save_step, 'summary save step')
tf.app.flags.DEFINE_integer('model_save_step', config.TRAIN.model_save_step, 'model save step')
tf.app.flags.DEFINE_integer('Samples_number', config.TRAIN.Samples_number, 'Samples number')
tf.app.flags.DEFINE_integer('Samples_Grid_X', config.TRAIN.Samples_Grid_X, 'Samples Grid Y')
tf.app.flags.DEFINE_integer('Samples_Grid_Y', config.TRAIN.Samples_Grid_Y, 'Samples Grid Y')
tf.app.flags.DEFINE_float('learning_rate_dis', config.TRAIN.lr_dis, 'learning rate_dis')
tf.app.flags.DEFINE_float('learning_rate_gen', config.TRAIN.lr_gen, 'learning rate_gen')
tf.app.flags.DEFINE_float('beta1', config.TRAIN.beta1, 'momentum term of Adam')
tf.app.flags.DEFINE_string('directory', config.PRE_TRAIN.directory,'Image directory')
tf.app.flags.DEFINE_string('directory_read', config.PRE_TRAIN.directory_read,'directory_read')
tf.app.flags.DEFINE_string('big_img_directory', config.PRE_TRAIN.big_img_directory,'big_img_directory')
tf.app.flags.DEFINE_string('output_directory', config.PRE_TRAIN.output_directory,'output directory')
tf.app.flags.DEFINE_integer('num_shards', config.PRE_TRAIN.shards,'Number of shards in the TFRecord files.')
tf.app.flags.DEFINE_integer('num_threads', config.PRE_TRAIN.threads,'Number of threads to preprocess the images.')
tf.app.flags.DEFINE_string('import_colorspace', config.PRE_TRAIN.import_colorspace,'import_colorspace of the images.')
tf.app.flags.DEFINE_integer('import_channels', config.PRE_TRAIN.import_channels,'import_channels of the images.')
tf.app.flags.DEFINE_string('file_name_shuffling', config.PRE_TRAIN.file_name_shuffling,'file_name_shuffling for import images')
tf.app.flags.DEFINE_string('log_dir', config.LOG_LOCATION , 'log directory')
tf.app.flags.DEFINE_string('checkpoint_dir', config.CHECKPOINT_LOCATION , 'checkpoint directory')
tf.app.flags.DEFINE_integer('train_steps', config.TRAIN.steps, 'number of train steps')
