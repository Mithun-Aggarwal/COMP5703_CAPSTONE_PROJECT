#This is the OPS file responsible to presetiing the layers required in any network
#Author - magg5201
#Created for Capstone Project COMP 5703

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

#The changed Conv_2d
#The parameters can be change from both FLAG and Config
def conv_2d(x, num_filters, 
            kernel_size=FLAGS.conv_2d_kernal, 
            stride=FLAGS.conv_2d_strides, scope='conv'):

    with tf.variable_scope(scope):
        w = tf.get_variable(
            'w', [kernel_size, kernel_size, x.get_shape()[-1], num_filters],
            initializer=tf.truncated_normal_initializer(stddev=FLAGS.conv_2d_stddev))

        conv = tf.nn.conv2d(
            x, w, strides=[1, stride, stride, 1], padding='SAME')

        biases = tf.get_variable(
            'biases', [num_filters], initializer=tf.constant_initializer(0.0))

        conv = tf.nn.bias_add(conv, biases)                   
        return conv


def conv2d_transpose(x,
                     output_shape,
                     kernel_size=5,
                     stride=2,
                     scope="conv_transpose"):

    with tf.variable_scope(scope):
        w = tf.get_variable(
            'w',
            [kernel_size, kernel_size, output_shape[-1], x.get_shape()[-1]],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        conv_transpose = tf.nn.conv2d_transpose(
            x, w, output_shape, strides=[1, stride, stride, 1])

        biases = tf.get_variable(
            'biases', [output_shape[-1]],
            initializer=tf.constant_initializer(0.0))

        conv_transpose = tf.nn.bias_add(conv_transpose, biases)

        return conv_transpose


def fc(x, num_outputs, scope="fc"):

    with tf.variable_scope(scope):
        w = tf.get_variable(
            'w', [x.get_shape()[-1], num_outputs],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        biases = tf.get_variable(
            'biases', [num_outputs], initializer=tf.constant_initializer(0.0))

        output = tf.nn.bias_add(tf.matmul(x, w), biases)

        return output


def batch_norm(x,
               decay=0.9,
               epsilon=1e-5,
               scale=True,
               is_training=True,
               reuse=False,
               scope='batch_norm'):

    bn = tf.contrib.layers.batch_norm(
        x,
        decay=decay,
        updates_collections=None,
        epsilon=epsilon,
        scale=scale,
        is_training=is_training,
        reuse=reuse,
        scope=scope)
    return bn


def leaky_relu(x, leak=0.2):
    return tf.maximum(x, leak * x)


from math import sqrt

def put_kernels_on_grid (kernel, pad = 1):

  '''Visualize conv. filters as an image (mostly for the 1st layer).
  Arranges filters into a grid, with some paddings between adjacent filters.
  Args:
    kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
    pad:               number of black pixels around each filter (between them)
  Return:
    Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
  '''
  # get shape of the grid. NumKernels == grid_Y * grid_X
  def factorization(n):
    for i in range(int(sqrt(float(n))), 0, -1):
      if n % i == 0:
        if i == 1: print('Who would enter a prime number of filters')
        return (i, int(n / i))
  (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)
  #print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

  x_min = tf.reduce_min(kernel)
  x_max = tf.reduce_max(kernel)
  kernel = (kernel - x_min) / (x_max - x_min)

  # pad X and Y
  x = tf.pad(kernel, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

  # X and Y dimensions, w.r.t. padding
  Y = kernel.get_shape()[0] + 2 * pad
  X = kernel.get_shape()[1] + 2 * pad

  channels = kernel.get_shape()[2]

  # put NumKernels to the 1st dimension
  x = tf.transpose(x, (3, 0, 1, 2))
  # organize grid on Y axis
  x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))

  # switch X and Y axes
  x = tf.transpose(x, (0, 2, 1, 3))
  # organize grid on X axis
  x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

  # back to normal order (not combining with the next step for clarity)
  x = tf.transpose(x, (2, 1, 3, 0))

  # to tf.image_summary order [batch_size, height, width, channels],
  #   where in this case batch_size == 1
  x = tf.transpose(x, (3, 0, 1, 2))

  # scaling to [0, 255] is not necessary for tensorboard
  return tf.image.convert_image_dtype(x, dtype = tf.uint8)
  #return x