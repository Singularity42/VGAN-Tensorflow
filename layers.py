import tensorflow as tf

def conv2d(input, filter_shape, strides, padding, stddev=0.02,
       name="conv2d"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', filter_shape,
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input, w, strides=strides, padding=padding)

    biases = tf.get_variable('biases', [filter_shape[-1]], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv

def deconv2d(input, filter_shape, output_shape,
       strides, padding, stddev=0.02,
       name="deconv2d", with_w=False):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', filter_shape,
              initializer=tf.random_normal_initializer(stddev=stddev))
    
    deconv = tf.nn.conv2d_transpose(input, w, output_shape=output_shape,
              strides=strides,padding=padding)

    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    return deconv

def conv3d(input, filter_shape, 
       strides,padding, stddev=0.02,
       name="conv3d",reuse=None):
  with tf.variable_scope(name,reuse=reuse):
    w = tf.get_variable('w', filter_shape,
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv3d(input, w, strides, padding=padding)

    biases = tf.get_variable('biases', [filter_shape[-1]], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv

def deconv3d(input, filter_shape, output_shape,
       strides, padding, stddev=0.02,
       name="deconv3d", with_w=False):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', filter_shape,
              initializer=tf.random_normal_initializer(stddev=stddev))
    
    deconv = tf.nn.conv3d_transpose(input, w, output_shape=output_shape,
              strides=strides,padding=padding)

    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
      return deconv,w
    else:
      return deconv


def batch_norm(input, scope, is_training=True,eps=1e-5,reuse=None):
  return tf.contrib.layers.batch_norm(input,
                      decay=0.1, 
                      updates_collections=None,
                      epsilon=eps,
                      scale=True,
                      is_training=is_training,
                      scope=scope,
                      reuse=reuse)

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def linear(x, output_dim, b):

    w = tf.get_variable("w", [x.get_shape()[1], output_dim])
    if b is None:
        b = tf.get_variable("b", [output_dim], intializer = tf.constant_initializer(0.0))
    
    return tf.matmul(x,w) + b

