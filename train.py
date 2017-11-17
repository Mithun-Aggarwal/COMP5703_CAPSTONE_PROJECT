# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 22:35:32 2017

@author: magg5201
For COMP 5703 - Capstone project 
Unsupervised learning with Gan for LF data

This the the execution file.
responsible for handling traing and session managemnt
"""

#import modules
import datetime
import os
import sys
import numpy as np
import tensorflow as tf

#framework modules
import dcgan
from config import config
import utils
import process_LF

FLAGS = tf.app.flags.FLAGS

#Function creaying various summaries to be displayted on Tensorboard
def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

#Fuction responsible for training the network
def train():
    z = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.z_dim], name='z')
    with tf.name_scope("Input"):
        images,label = dcgan.inputs(batch_size=FLAGS.batch_size)
        with tf.name_scope("Input_image"):
            tf.summary.image('Input Image',images,FLAGS.batch_size,\
                             collections=["Input_image"])
    # logits
    with tf.name_scope("logits"):
        D_logits_real, D_logits_fake, generated_images = dcgan.inference(images, z)
        with tf.name_scope("D_logits_real"):
            variable_summaries(D_logits_fake)
        with tf.name_scope("D_logits_real"):
            variable_summaries(D_logits_fake)
    
    #images to tensorboard    
    with tf.name_scope("Generated_image"):
        tf.summary.image('Generated Image',generated_images,FLAGS.batch_size,\
                         collections=["Generated_image"])

          
    # loss
    with tf.name_scope("loss"):
        d_loss, g_loss, d_loss_real , d_loss_fake = dcgan.loss(D_logits_real, D_logits_fake)
        with tf.name_scope("d_loss"):
            variable_summaries(d_loss)
        with tf.name_scope("g_loss"):
            variable_summaries(g_loss)
        with tf.name_scope("d_loss_real"):
            variable_summaries(d_loss_real)
        with tf.name_scope("d_loss_fake"):
            variable_summaries(d_loss_fake) 
            
    # train the model
    with tf.name_scope("train_ops"):
        train_d_op, train_g_op = dcgan.train(d_loss, g_loss)
            
    #input filter to tensorboard
    dcgan.addfilterstensorboard(FLAGS.num_filters)
    
    #Variable to be initialized
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    
    #Runtime options
    summary_op = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)
    summary_op_Generated_image = tf.summary.merge_all(key="Generated_image")
    summary_op_Input_image = tf.summary.merge_all(key="Input_image")
    summary_op_Generated_filters = tf.summary.merge_all(key="Generated_filters")
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    
    #GPU_&_Computation
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_use)
    coord = tf.train.Coordinator()
    if (FLAGS.restore_model == "False"):
        saver = tf.train.Saver()
    if (FLAGS.restore_model == "True"):
# =============================================================================
#             saver = tf.train.Saver()
#             saver.restore(sess, str("~/model.ckp.index"))
#             print("Model restored.")
# =============================================================================
        tf.reset_default_graph()  
        #imported_meta = tf.train.import_meta_graph("~/model.ckp.meta")        
    
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        summary_writer = tf.summary.FileWriter(config.LOG_LOCATION,sess.graph)
        sess.run(init)
        threads = tf.train.start_queue_runners(sess=sess,coord = coord)
        
        #if (FLAGS.restore_model == "True"):
            #imported_meta.restore(sess, tf.train.latest_checkpoint("~/LF_DCGAN/checkpoints/LF_DCGAN_377_547/checkpoint'))
        training_steps = FLAGS.train_steps

        for step in range(training_steps):
            #print('global_step: %s' % tf.train.global_step(sess, global_step_tensor))
            random_z = np.random.uniform(
                -1, 1, size=(FLAGS.batch_size, FLAGS.z_dim)).astype(np.float32)

            if ( step == 1):
                num = FLAGS.extra_Training_discriminator
                while num != 0:
                    sess.run(train_d_op, feed_dict={z: random_z})
                    num = num -1 
                    print("discriminator training : " + str(num))
            sess.run(train_d_op, feed_dict={z: random_z})
            sess.run(train_g_op, feed_dict={z: random_z})
            sess.run(train_g_op, feed_dict={z: random_z})
            with tf.name_scope("Loss"):
                discrimnator_loss, generator_loss = sess.run(
                    [d_loss, g_loss], feed_dict={z: random_z})
                with tf.name_scope("discrimnator_loss"):
                    variable_summaries(discrimnator_loss)
                with tf.name_scope("generator_loss"):
                    variable_summaries(generator_loss)
                
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, d_loss {:g}, g_loss {:g}".format(
                time_str, step, discrimnator_loss, generator_loss))
            
            summary_writer.add_run_metadata(run_metadata, 'step%03d' % step)
            
            summary = sess.run(summary_op, feed_dict={z: random_z},
                               options=run_options,
                               run_metadata=run_metadata)
            summary_writer.add_summary(summary,step) 

            if step % FLAGS.summary_save_step == 0:
                summary_Generated_image = sess.run(summary_op_Generated_image,feed_dict={z: random_z})
                summary_Generated_filters = sess.run(summary_op_Generated_filters)
                summary_Input_image = sess.run(summary_op_Input_image)
                summary_writer.add_summary(summary_Generated_image,step)
                summary_writer.add_summary(summary_Generated_filters,step)
                summary_writer.add_summary(summary_Input_image,step)   
            
            if step % FLAGS.model_save_step == 0:
                test_images = sess.run(generated_images, feed_dict={z: random_z})          
                image_path = os.path.join(FLAGS.checkpoint_dir,
                                      "sampled_images_%d.png" % step)
                utils.grid_plot(test_images, [FLAGS.Samples_Grid_X, FLAGS.Samples_Grid_Y], image_path)
                saver.save(sess, os.path.join(FLAGS.checkpoint_dir, "model.ckp"))
    coord.request_stop()
    coord.join(threads)
    summary_writer.close()
    
def main(argv=None):

    if (FLAGS.restore_model == "True"):
        train()
    
    if (FLAGS.imgprocessing_status == "True"):
        if tf.gfile.Exists(FLAGS.directory):
            tf.gfile.DeleteRecursively(FLAGS.directory)
        tf.gfile.MakeDirs(FLAGS.directory)
        if tf.gfile.Exists(FLAGS.test_path):
            tf.gfile.DeleteRecursively(FLAGS.test_path)
        tf.gfile.MakeDirs(FLAGS.test_path)
        CUT_DIM = [FLAGS.CUT_DIM_X,FLAGS.CUT_DIM_Y]
        utils.ont_to_64(FLAGS.big_img_directory,FLAGS.directory,CUT_DIM,FLAGS.CUT_X,FLAGS.CUT_Y,FLAGS.test_path)
        
    if (FLAGS.TF_LF_creation == "True"):
        print("1-64 complete - starting to change the records in TF")
        if tf.gfile.Exists(FLAGS.output_directory):
            tf.gfile.DeleteRecursively(FLAGS.output_directory)
        tf.gfile.MakeDirs(FLAGS.output_directory)        
        process_LF._process_dataset("LF", FLAGS.directory_read, FLAGS.num_shards)
        
    if (FLAGS.Train_status == "True"):
        if tf.gfile.Exists(FLAGS.log_dir):
            tf.gfile.DeleteRecursively(FLAGS.log_dir)
        tf.gfile.MakeDirs(FLAGS.log_dir)
        
        if tf.gfile.Exists(FLAGS.checkpoint_dir):
            tf.gfile.DeleteRecursively(FLAGS.checkpoint_dir)
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)
        train()
        
    else:
        sys.exit()
    
if __name__ == '__main__':
    tf.app.run()
