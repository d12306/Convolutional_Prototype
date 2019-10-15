from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math
slim = tf.contrib.slim


def densenet_arg_scope(weight_decay=0.0001,
                       batch_norm_decay=0.997,
                       batch_norm_epsilon=1e-5,
                       batch_norm_scale=True):
    """Defines the default DenseNet arg scope.
    Args:
      weight_decay: The weight decay to use for regularizing the model.
      batch_norm_decay: The moving average decay when estimating layer activation
        statistics in batch normalization.
      batch_norm_epsilon: Small constant to prevent division by zero when
        normalizing activations by their variance in batch normalization.
      batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
        activations in the batch normalization layer.
    Returns:
      An `arg_scope` to use for the densenet models.
    """
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=tf.nn.relu,
    ):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc


def densenet_bc(inputs,
                num_classes=None,
                is_training=True,
                growth_rate=12,
                drop_rate=0,
                depth=100,
                for_imagenet=False,
                reuse=None,
                scope=None):
    """Generator for DenseNet models.
    Args:
      inputs: A tensor of size [batch, height_in, width_in, channels].
      num_classes: Number of predicted classes for classification tasks. If None
        we return the features before the logit layer.
      is_training: whether is training or not.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.
    Returns:
      net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      end_points: A dictionary from components of the network to the corresponding
        activation.
    Raises:
      ValueError: If the target output_stride is not valid.
    """
    n_channels = 2 * growth_rate
    reduction = 0.5
    bottleneck = True
    N = int((depth - 4) / (6 if bottleneck else 3))

    def single_layer(input, n_out_channels, drop_rate):

        conv = slim.batch_norm(input, activation_fn=tf.nn.relu)
        conv = slim.conv2d(conv, n_out_channels, [3, 3], stride=[1, 1])
        if drop_rate > 0:
            conv = tf.nn.dropout(conv, drop_rate)

        out = tf.concat(3, [tf.identity(input), conv])

        return out

    def bottleneck_layer(input, n_output_channels, drop_rate):

        inter_channels = 4 * n_output_channels

        conv = slim.batch_norm(input, activation_fn=tf.nn.relu)
        conv = slim.conv2d(conv, inter_channels, [1, 1], stride=[1, 1])
        if drop_rate > 0:
            conv = tf.nn.dropout(conv, drop_rate)

        conv = slim.batch_norm(conv, activation_fn=tf.nn.relu)
        conv = slim.conv2d(conv, n_output_channels, [3, 3], stride=[1, 1])
        if drop_rate > 0:
            conv = tf.nn.dropout(conv, drop_rate)

        out = tf.concat([tf.identity(input), conv], axis = 3)

        return out

    if bottleneck:
        add = bottleneck_layer
    else:
        add = single_layer

    def transition(input, n_output_channels, drop_rate):

        conv = slim.batch_norm(input, activation_fn=tf.nn.relu)
        conv = slim.conv2d(conv, n_output_channels, [1, 1], stride=[1, 1])
        if drop_rate > 0:
            conv = tf.nn.dropout(conv, drop_rate)
        conv = slim.avg_pool2d(conv, [2, 2], stride=2)

        return conv

    with tf.variable_scope(scope, 'densenet', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.avg_pool2d],
                            outputs_collections=end_points_collection, padding='SAME'):
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                net = inputs

                if for_imagenet:
                    net = slim.conv2d(net, n_channels, [7, 7], stride=2)
                    net = slim.max_pool2d(net, [3, 3], stride=2)

                    for i in range(0, 6):
                        net = add(net, growth_rate, drop_rate)
                        n_channels += growth_rate

                    net = transition(net, math.floor(n_channels * reduction), drop_rate)
                    n_channels = math.floor(n_channels * reduction)

                    for i in range(0, 12):
                        net = add(net, growth_rate, drop_rate)
                        n_channels += growth_rate

                    net = transition(net, math.floor(n_channels * reduction), drop_rate)
                    n_channels = math.floor(n_channels * reduction)

                    for i in range(0, 36):
                        net = add(net, growth_rate, drop_rate)
                        n_channels += growth_rate

                    net = transition(net, math.floor(n_channels * reduction), drop_rate)
                    n_channels = math.floor(n_channels * reduction)

                    for i in range(0, 24):
                        net = add(net, growth_rate, drop_rate)
                        n_channels += growth_rate

                    net = slim.batch_norm(net, activation_fn=tf.nn.relu)
                    net = slim.avg_pool2d(net, [7, 7], stride=7)

                else:
                    net = slim.conv2d(net, n_channels, [3, 3], stride=1)

                    for i in range(0, N):
                        net = add(net, growth_rate, drop_rate)
                        n_channels += growth_rate

                    net = transition(net, math.floor(n_channels * reduction), drop_rate)
                    n_channels = math.floor(n_channels * reduction)

                    for i in range(0, N):
                        net = add(net, growth_rate, drop_rate)
                        n_channels += growth_rate

                    net = transition(net, math.floor(n_channels * reduction), drop_rate)
                    n_channels = math.floor(n_channels * reduction)

                    for i in range(0, N):
                        net = add(net, growth_rate, drop_rate)
                        n_channels += growth_rate

                    net = slim.batch_norm(net, activation_fn=tf.nn.relu)
                    net_feature = slim.avg_pool2d(net, [8, 8], stride=8)

                net_feature = tf.reshape(net_feature, [tf.shape(net_feature)[0],\
                   342 ])
                # import ipdb
                # ipdb.set_trace()
                # tf.shape(net_feature)[1]*tf.shape(net_feature)[2]*tf.shape(net_feature)[3]
                if num_classes is not None:
                    net = slim.fully_connected(net_feature, num_classes, scope='fc/fc_2')
                    # net = slim.conv2d(net_feature, num_classes, [1, 1], activation_fn=None,
                    #                   normalizer_fn=None, scope='logits')
                # net = tf.squeeze(net)
                # Convert end_points_collection into a dictionary of end_points.
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                if num_classes is not None:
                    end_points['predictions'] = slim.softmax(net, scope='predictions')


                return net_feature, net


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math


# __all__ = ['densenet']


# from torch.autograd import Variable

# class Bottleneck(nn.Module):
#     def __init__(self, inplanes, expansion=4, growthRate=12, dropRate=0):
#         super(Bottleneck, self).__init__()
#         planes = expansion * growthRate
#         self.bn1 = nn.BatchNorm2d(inplanes)
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, growthRate, kernel_size=3, 
#                                padding=1, bias=False)
#         self.relu = nn.ReLU(inplace=True)
#         self.dropRate = dropRate

#     def forward(self, x):
#         out = self.bn1(x)
#         out = self.relu(out)
#         out = self.conv1(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         if self.dropRate > 0:
#             out = F.dropout(out, p=self.dropRate, training=self.training)

#         out = torch.cat((x, out), 1)

#         return out


# class BasicBlock(nn.Module):
#     def __init__(self, inplanes, expansion=1, growthRate=12, dropRate=0):
#         super(BasicBlock, self).__init__()
#         planes = expansion * growthRate
#         self.bn1 = nn.BatchNorm2d(inplanes)
#         self.conv1 = nn.Conv2d(inplanes, growthRate, kernel_size=3, 
#                                padding=1, bias=False)
#         self.relu = nn.ReLU(inplace=True)
#         self.dropRate = dropRate

#     def forward(self, x):
#         out = self.bn1(x)
#         out = self.relu(out)
#         out = self.conv1(out)
#         if self.dropRate > 0:
#             out = F.dropout(out, p=self.dropRate, training=self.training)

#         out = torch.cat((x, out), 1)

#         return out


# class Transition(nn.Module):
#     def __init__(self, inplanes, outplanes):
#         super(Transition, self).__init__()
#         self.bn1 = nn.BatchNorm2d(inplanes)
#         self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1,
#                                bias=False)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         out = self.bn1(x)
#         out = self.relu(out)
#         out = self.conv1(out)
#         out = F.avg_pool2d(out, 2)
#         return out


# class DenseNet(nn.Module):

#     def __init__(self, depth=22, block=Bottleneck, 
#         dropRate=0, num_classes=10, growthRate=12, compressionRate=2):
#         super(DenseNet, self).__init__()

#         assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
#         n = (depth - 4) / 3 if block == BasicBlock else (depth - 4) // 6

#         self.growthRate = growthRate
#         self.dropRate = dropRate

#         # self.inplanes is a global variable used across multiple
#         # helper functions
#         self.inplanes = growthRate * 2 
#         self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
#                                bias=False)
#         self.dense1 = self._make_denseblock(block, n)
#         self.trans1 = self._make_transition(compressionRate)
#         self.dense2 = self._make_denseblock(block, n)
#         self.trans2 = self._make_transition(compressionRate)
#         self.dense3 = self._make_denseblock(block, n)
#         self.bn = nn.BatchNorm2d(self.inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         self.avgpool = nn.AvgPool2d(8)
#         self.fc = nn.Linear(self.inplanes, num_classes)

#         # Weight initialization
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

#     def _make_denseblock(self, block, blocks):
#         layers = []
#         for i in range(blocks):
#             # Currently we fix the expansion ratio as the default value
#             layers.append(block(self.inplanes, growthRate=self.growthRate, dropRate=self.dropRate))
#             self.inplanes += self.growthRate

#         return nn.Sequential(*layers)

#     def _make_transition(self, compressionRate):
#         inplanes = self.inplanes
#         outplanes = int(math.floor(self.inplanes // compressionRate))
#         self.inplanes = outplanes
#         return Transition(inplanes, outplanes)


#     def forward(self, x):
#         x = self.conv1(x)

#         x = self.trans1(self.dense1(x)) 
#         x = self.trans2(self.dense2(x)) 
#         x = self.dense3(x)
#         x = self.bn(x)
#         x = self.relu(x)

#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)

#         return x


# def densenet(**kwargs):
#     """
#     Constructs a ResNet model.
#     """
#     return DenseNet(**kwargs)