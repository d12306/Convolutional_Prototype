import os
import numpy as np
import pickle

def load_cifar_datafile(filename):
  import pickle
  with open(filename, 'rb') as fo:
      # if version.major == 3:
      data_dict = pickle.load(fo, encoding='bytes')
      # else:
      #     data_dict = pickle.load(fo)
      assert data_dict[b'data'].dtype == np.uint8
      image_data = data_dict[b'data']
      image_data = image_data.reshape(
          (10000, 3, 32, 32)).transpose(0, 2, 3, 1)

      image_data = np.array(image_data, dtype = np.int32)
      labels = np.array(data_dict[b'labels'], dtype = np.int32)
      return image_data ,labels


def load_cifar():
    filename_1 = os.path.join('/home/xfdu/OVA-master/cifar10/OVA-master/cifar10_challenge/cifar10_data/', 'data_batch_1')
    filename_2 = os.path.join('/home/xfdu/OVA-master/cifar10/OVA-master/cifar10_challenge/cifar10_data/', 'data_batch_2')
    filename_3 = os.path.join('/home/xfdu/OVA-master/cifar10/OVA-master/cifar10_challenge/cifar10_data/', 'data_batch_3')
    filename_4 = os.path.join('/home/xfdu/OVA-master/cifar10/OVA-master/cifar10_challenge/cifar10_data/', 'data_batch_4')
    filename_5 = os.path.join('/home/xfdu/OVA-master/cifar10/OVA-master/cifar10_challenge/cifar10_data/', 'data_batch_5')
    filename_test = os.path.join('/home/xfdu/OVA-master/cifar10/OVA-master/cifar10_challenge/cifar10_data/', 'test_batch')


    Xtest_1, Ytest_1 = load_cifar_datafile(filename_1)
    Xtest_2, Ytest_2 = load_cifar_datafile(filename_2)
    Xtest_3, Ytest_3 = load_cifar_datafile(filename_3)
    Xtest_4, Ytest_4 = load_cifar_datafile(filename_4)
    Xtest_5, Ytest_5 = load_cifar_datafile(filename_5)

    Xtest_test, Ytest_test = load_cifar_datafile(filename_test)

    Xtrain_all = Xtest_1/255.
    Xtrain_all = np.concatenate((Xtrain_all, Xtest_2/255.),axis = 0)
    Xtrain_all = np.concatenate((Xtrain_all, Xtest_3/255.),axis = 0)
    Xtrain_all = np.concatenate((Xtrain_all, Xtest_4/255.),axis = 0)
    Xtrain_all = np.concatenate((Xtrain_all, Xtest_5/255.),axis = 0)

    Ytrain_all = Ytest_1
    Ytrain_all = np.concatenate((Ytrain_all, Ytest_2),axis = 0)
    Ytrain_all = np.concatenate((Ytrain_all, Ytest_3),axis = 0)
    Ytrain_all = np.concatenate((Ytrain_all, Ytest_4),axis = 0)
    Ytrain_all = np.concatenate((Ytrain_all, Ytest_5),axis = 0)

    Xtest_test = Xtest_test /255.
    Ytest_test = Ytest_test

    return Xtrain_all, Ytrain_all, Xtest_test, Ytest_test


# def load_cifar():
#     filename_1 = os.path.join('/home/xfdu/OVA-master/cifar10/OVA-master/cifar10_challenge/cifar10_data/', 'data_batch_1')
#     filename_2 = os.path.join('/home/xfdu/OVA-master/cifar10/OVA-master/cifar10_challenge/cifar10_data/', 'data_batch_2')
#     filename_3 = os.path.join('/home/xfdu/OVA-master/cifar10/OVA-master/cifar10_challenge/cifar10_data/', 'data_batch_3')
#     filename_4 = os.path.join('/home/xfdu/OVA-master/cifar10/OVA-master/cifar10_challenge/cifar10_data/', 'data_batch_4')
#     filename_5 = os.path.join('/home/xfdu/OVA-master/cifar10/OVA-master/cifar10_challenge/cifar10_data/', 'data_batch_5')
#     filename_test = os.path.join('/home/xfdu/OVA-master/cifar10/OVA-master/cifar10_challenge/cifar10_data/', 'test_batch')


#     Xtest_1, Ytest_1 = load_cifar_datafile(filename_1)
#     Xtest_2, Ytest_2 = load_cifar_datafile(filename_2)
#     Xtest_3, Ytest_3 = load_cifar_datafile(filename_3)
#     Xtest_4, Ytest_4 = load_cifar_datafile(filename_4)
#     Xtest_5, Ytest_5 = load_cifar_datafile(filename_5)

#     Xtest_test, Ytest_test = load_cifar_datafile(filename_test)

#     Xtrain_all = Xtest_1
#     Xtrain_all = np.concatenate((Xtrain_all, Xtest_2),axis = 0)
#     Xtrain_all = np.concatenate((Xtrain_all, Xtest_3),axis = 0)
#     Xtrain_all = np.concatenate((Xtrain_all, Xtest_4),axis = 0)
#     Xtrain_all = np.concatenate((Xtrain_all, Xtest_5),axis = 0)

#     Ytrain_all = Ytest_1
#     Ytrain_all = np.concatenate((Ytrain_all, Ytest_2),axis = 0)
#     Ytrain_all = np.concatenate((Ytrain_all, Ytest_3),axis = 0)
#     Ytrain_all = np.concatenate((Ytrain_all, Ytest_4),axis = 0)
#     Ytrain_all = np.concatenate((Ytrain_all, Ytest_5),axis = 0)

#     Xtest_test = Xtest_test 
#     Ytest_test = Ytest_test

#     return Xtrain_all, Ytrain_all, Xtest_test, Ytest_test


class Model_Resnet(object):
  """ResNet model."""

  def __init__(self, class_num):
    """ResNet constructor.
    Args:
      mode: One of 'train' and 'eval'.
    """
    # self.mode = mode
    # self.digit = digit
    self.class_num = class_num
    # self.build_model()

  def add_internal_summaries(self):
    pass

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  def build_model(self, x, mode):
    self.mode = mode
    assert self.mode == 'train' or self.mode == 'eval'
    """Build the core model within the graph."""

    with tf.variable_scope('input'):

      self.x_input = x
      input_standardized = tf.map_fn(lambda img: tf.image.per_image_standardization(img),
                               self.x_input)
      x = self._conv('init_conv', self.x_input, 3, 3, 16, self._stride_arr(1))



    strides = [1, 2, 2]
    activate_before_residual = [True, False, False]
    res_func = self._residual

    # Uncomment the following codes to use w28-10 wide residual network.
    # It is more memory efficient than very deep residual network and has
    # comparably good performance.
    # https://arxiv.org/pdf/1605.07146v1.pdf
    filters = [16, 160, 320, 640]
    # filters = [16, 32, 64, 128]



    # Update hps.num_residual_units to 9

    with tf.variable_scope('unit_1_0'):
      x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
                   activate_before_residual[0])
    for i in range(1, 5):
      with tf.variable_scope('unit_1_%d' % i):
        x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

    with tf.variable_scope('unit_2_0'):
      x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                   activate_before_residual[1])
    for i in range(1, 5):
      with tf.variable_scope('unit_2_%d' % i):
        x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

    with tf.variable_scope('unit_3_0'):
      x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                   activate_before_residual[2])
    for i in range(1, 5):
      with tf.variable_scope('unit_3_%d' % i):
        x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)

    with tf.variable_scope('unit_last'):
      x = self._batch_norm('final_bn', x)
      x = self._relu(x, 0.1)
      x_feature = self._global_avg_pool(x)

    with tf.variable_scope('logit'):
      self.pre_softmax = self._fully_connected(x_feature, self.class_num)
    self.weight_decay_loss = self._decay()

    return x_feature, self.pre_softmax

    # self.predictions = tf.argmax(self.pre_softmax, 1)
    # self.correct_prediction = tf.equal(self.predictions, self.y_input)
    # self.num_correct = tf.reduce_sum(
    #     tf.cast(self.correct_prediction, tf.int64))
    # self.accuracy = tf.reduce_mean(
    #     tf.cast(self.correct_prediction, tf.float32))

    # with tf.variable_scope('costs'):
    #   self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #       logits=self.pre_softmax, labels=self.y_input)
    #   self.xent = tf.reduce_sum(self.y_xent, name='y_xent')
    #   self.mean_xent = tf.reduce_mean(self.y_xent)
    

  def _batch_norm(self, name, x):
    """Batch normalization."""
    with tf.name_scope(name):
      return tf.contrib.layers.batch_norm(
          inputs=x,
          decay=.9,
          center=True,
          scale=True,
          activation_fn=None,
          updates_collections=None,
          is_training=(self.mode == 'train'))

  def _residual(self, x, in_filter, out_filter, stride,
                activate_before_residual=False):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('shared_activation'):
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, 0.1)
        orig_x = x
    else:
      with tf.variable_scope('residual_only_activation'):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, 0.1)

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

    with tf.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, 0.1)
      x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
        orig_x = tf.pad(
            orig_x, [[0, 0], [0, 0], [0, 0],
                     [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
      x += orig_x

    tf.logging.debug('image after unit %s', x.get_shape())
    return x

  def _decay(self):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find('weights') > 0:
        costs.append(tf.nn.l2_loss(var))
    return tf.add_n(costs)

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      kernel = tf.get_variable(
          'weights', [filter_size, filter_size, in_filters, out_filters],
          tf.float32, initializer=tf.random_normal_initializer(
              stddev=np.sqrt(2.0/n)))
      return tf.nn.conv2d(x, kernel, strides, padding='SAME')

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    num_non_batch_dimensions = len(x.shape)
    prod_non_batch_dimensions = 1
    for ii in range(num_non_batch_dimensions - 1):
      prod_non_batch_dimensions *= int(x.shape[ii + 1])
    x = tf.reshape(x, [tf.shape(x)[0], -1])
    w = tf.get_variable(
        'weights', [prod_non_batch_dimensions, out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b)

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])
