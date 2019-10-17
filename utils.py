import os
# import numpy as np
import numpy as np
import tensorflow as tf
import pickle


class data_augmentor(object):
    """docstring for data_augmentor"""
    def __init__(self, x):
        super(data_augmentor, self).__init__()

        self.x = x
        # self.x = tf.image.random_hue(self.x, 0.08)
        # self.x = tf.image.random_saturation(self.x, 0.6, 1.6)
        # self.x = tf.image.random_brightness(self.x, 0.05)
        # self.x = tf.image.random_contrast(self.x, 0.7, 1.3) 
        # self.x = tf.image.per_image_standardization(self.x)
        padded = tf.map_fn(lambda img: tf.image.resize_image_with_crop_or_pad(
            img, 32 + 4, 32 + 4),
            self.x)
        cropped = tf.map_fn(lambda img: tf.random_crop(img, [32,
                                                             32,
                                                             3]), padded)
        flipped = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), cropped)
        # flipped = tf.map_fn(lambda img: tf.image.random_flip_up_down(img), flipped)
        self.augmented = flipped
        # self.augmented = tf.image.rot90(self.augmented)

    def output(self, sess, x_train):
        return sess.run(self.augmented, feed_dict={self.x: x_train})


def load_cifar_datafile(filename):
  import pickle
  with open(filename, 'rb') as fo:
      data_dict = pickle.load(fo, encoding='bytes')
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

def top_k_error(self, predictions, labels, k):
    '''
    Calculate the top-k error
    :param predictions: 2D tensor with shape [batch_size, num_labels]
    :param labels: 1D tensor with shape [batch_size, 1]
    :param k: int
    :return: tensor with shape [1]
    '''
    batch_size = predictions.get_shape().as_list()[0]
    in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))
    num_correct = tf.reduce_sum(in_top1)
    return (batch_size - num_correct) / float(batch_size)


'''
This is the resnet structure
'''
BN_EPSILON = 1e-5
#A small float number to avoid dividing by 0.

def activation_summary(x):
    '''
    :param x: A Tensor
    :return: Add histogram summary and scalar summary of the sparsity of the tensor
    '''
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def create_variables(name, shape, weight_decay,initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
    # tf.contrib.layers.variance_scaling_initializer()
    # tf.contrib.layers.xavier_initializer()
    # tf.initializers.truncated_normal()
    # tf.random_normal_initializer()
    # tf.glorot_normal_initializer()
    # tf.glorot_uniform_initializer()
    # tf.keras.initializers.he_normal()
    # tf.uniform_unit_scaling_initializer(factor=1.0)
    '''
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
    layers.
    :return: The created variable
    '''
    ## TODO: to allow different weight decay to fully connected layer and conv layer
    regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)
    new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                    regularizer=regularizer)
    return new_variables


def output_layer(input_layer, num_labels, weight_decay):
    '''
    :param input_layer: 2D tensor
    :param num_labels: int. How many output labels in total? (10 for cifar10 and 100 for cifar100)
    :return: output layer Y = WX + B
    '''
    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name='fc_weights', weight_decay = weight_decay, shape=[input_dim, num_labels], is_fc_layer=True,
                            initializer=tf.contrib.layers.xavier_initializer())
                            # initializer=tf.contrib.layers.variance_scaling_initializer())
                            # initializer = tf.keras.initializers.he_normal())
    fc_b = create_variables(name='fc_bias', weight_decay = weight_decay, shape=[num_labels], initializer=tf.zeros_initializer())
    fc_h = tf.matmul(input_layer, fc_w) + fc_b
    return fc_h


def batch_normalization_layer(input_layer, dimension):
    '''
    Helper function to do batch normalziation
    :param input_layer: 4D tensor
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
    :return: the 4D tensor after being normalized
    '''
    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    beta = tf.get_variable('beta', dimension, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma', dimension, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)
    return bn_layer

def conv_bn_relu_layer(input_layer, filter_shape, stride, weight_decay):
    '''
    A helper function to conv, batch normalize and relu the input tensor sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
    '''
    out_channel = filter_shape[-1]
    filter = create_variables(name='conv', weight_decay = weight_decay, shape=filter_shape)
    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    bn_layer = batch_normalization_layer(conv_layer, out_channel)

    output = tf.nn.relu(bn_layer)
    return output


def bn_relu_conv_layer(input_layer, filter_shape, stride, weight_decay):
    '''
    A helper function to batch normalize, relu and conv the input layer sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
    '''
    in_channel = input_layer.get_shape().as_list()[-1]

    bn_layer = batch_normalization_layer(input_layer, in_channel)
    relu_layer = tf.nn.relu(bn_layer)

    filter = create_variables(name='conv', weight_decay = weight_decay, shape=filter_shape)
    conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    return conv_layer



def residual_block(input_layer, output_channel, weight_decay, first_block=False):
    '''
    Defines a residual block in ResNet
    :param input_layer: 4D tensor
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param first_block: if this is the first residual block of the whole network
    :return: 4D tensor.
    '''
    input_channel = input_layer.get_shape().as_list()[-1]

    # When it's time to "shrink" the image size, we use stride = 2
    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    # The first conv layer of the first residual block does not need to be normalized and relu-ed.
    with tf.variable_scope('conv1_in_block'):
        if first_block:
            filter = create_variables(name='conv', weight_decay = weight_decay, shape=[3, 3, input_channel, output_channel])
            conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
        else:
            conv1 = bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride, weight_decay = weight_decay)

    with tf.variable_scope('conv2_in_block'):
        conv2 = bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], 1, weight_decay = weight_decay)

    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    #  depth of input layers
    if increase_dim is True:
        pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='VALID')
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                     input_channel // 2]])
    else:
        padded_input = input_layer

    output = conv2 + padded_input
    return output


def inference(input_tensor_batch, n, num_classes, reuse, weight_decay):
    '''
    The main function that defines the ResNet. total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
    :param input_tensor_batch: 4D tensor
    :param n: num_residual_blocks
    :param reuse: To build train graph, reuse=False. To build validation graph and share weights
    with train graph, resue=True
    :return: last layer in the network. Not softmax-ed
    '''

    layers = []
    with tf.variable_scope('conv0', reuse=reuse):
        conv0 = conv_bn_relu_layer(input_tensor_batch, [3, 3, 3, 16], 1, weight_decay)
        activation_summary(conv0)
        layers.append(conv0)

    for i in range(n):
        with tf.variable_scope('conv1_%d' %i, reuse=reuse):
            if i == 0:
                conv1 = residual_block(layers[-1], 16, weight_decay = weight_decay, first_block=True)
            else:
                conv1 = residual_block(layers[-1], 16, weight_decay = weight_decay)
            activation_summary(conv1)
            layers.append(conv1)

    for i in range(n):
        with tf.variable_scope('conv2_%d' %i, reuse=reuse):
            conv2 = residual_block(layers[-1], 32, weight_decay = weight_decay)
            activation_summary(conv2)
            layers.append(conv2)

    for i in range(n):
        with tf.variable_scope('conv3_%d' %i, reuse=reuse):
            conv3 = residual_block(layers[-1], 64, weight_decay = weight_decay)
            layers.append(conv3)
        assert conv3.get_shape().as_list()[1:] == [8, 8, 64]

    with tf.variable_scope('fc', reuse=reuse):
        in_channel = layers[-1].get_shape().as_list()[-1]
        bn_layer = batch_normalization_layer(layers[-1], in_channel)
        relu_layer = tf.nn.relu(bn_layer)
        global_pool = tf.reduce_mean(relu_layer, [1, 2])

        assert global_pool.get_shape().as_list()[-1:] == [64]
        output = output_layer(global_pool, num_classes, weight_decay = weight_decay)
        layers.append(output)

    return global_pool, layers[-1]


def test_graph(train_dir='logs'):
    '''
    Run this function to look at the graph structure on tensorboard. A fast way!
    :param train_dir:
    '''
    input_tensor = tf.constant(np.ones([128, 32, 32, 3]), dtype=tf.float32)
    result = inference(input_tensor, 2, reuse=False)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)

