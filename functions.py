#################################################
import numpy as np
import tensorflow as tf
# import cPickle as pickle
import struct    
import os
import matplotlib.pyplot as plt

def visualize(feat, labels, epoch, centers, args):
    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.clf()
    for i in range(10):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
        
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')

    for i in range(10):
        plt.plot(centers[i,:,0], centers[i,:,1], 'D', c=c[i])

    plt.text(-4.8, 4.6, "epoch=%d" % epoch)

    if not os.path.isdir('./images/'):  
        os.makedirs('./images/')
    plt.savefig('./images/'+args.loss+'_'+args.dataset+'_'+str(args.use_dot_product)+'_epoch=%d.jpg' % epoch, dpi = 250)
    plt.close()

##################################################
# compute distances between the sample features
# with the centers
def distance(features, centers, flags):

    features = tf.tile(tf.expand_dims(tf.expand_dims(features, axis = 1),axis = 1), multiples= [1,flags.num_classes,flags.num_protos,1])
    size = tf.shape(features)[0]
    centers = tf.tile(tf.expand_dims(centers, axis = 0), multiples = [size,1,1,1])
    dist = tf.norm(features - centers, axis = 3)
    # import ipdb
    # ipdb.set_trace()
    # f_2 = tf.reduce_sum(tf.pow(features, 2), axis=2, keep_dims=True)
    # c_2 = tf.reduce_sum(tf.pow(centers, 2), axis=2, keep_dims=True)
    # dist = f_2 - 2*tf.matmul(features, centers, transpose_b=True) + tf.transpose(c_2, perm=[1,0])
    return dist

def dot_product(features, centers, flags):
    features = tf.tile(tf.expand_dims(tf.expand_dims(features, axis = 1),axis = 1), multiples= [1,flags.num_classes,flags.num_protos,1])
    size = tf.shape(features)[0]
    centers = tf.tile(tf.expand_dims(centers, axis = 0), multiples = [size,1,1,1])
    product = tf.multiply(features, centers)
    product = tf.reduce_sum(product, axis = 3)

    return product

# the cross entorpy loss for the traditional 
# softmax layer based  neural networks
def softmax_loss(logits, labels):
    labels = tf.to_int32(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
        logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')

# L2 regular loss
def regular_loss(name):
    params = tf.get_collection(name)
    return tf.add_n([tf.nn.l2_loss(i) for i in params])

# margin based classification loss (MCL)
def mcl_loss(features, labels, centers, margin):
    dist = distance(features, centers)

    values, indexes = tf.nn.top_k(-dist, k=2, sorted=True)
    top2 = -values
    d_1 = top2[:, 0]
    d_2 = top2[:, 1]

    row_idx = tf.range(tf.shape(labels)[0], dtype=tf.int32)
    idx = tf.stack([row_idx, labels], axis=1)
    d_y = tf.gather_nd(dist, idx, name='dy')

    indicator = tf.cast(tf.nn.in_top_k(-dist, labels, k=1), tf.float32)
    d_c = indicator*d_2 + (1-indicator)*d_1

    loss = tf.nn.relu(d_y-d_c+margin, name='loss')
    mean_loss = tf.reduce_mean(loss, name='mean_loss')

    return mean_loss

# generalized margin based classification loss (GMCL)
def gmcl_loss(features, labels, centers, margin):
    dist = distance(features, centers)

    values, indexes = tf.nn.top_k(-dist, k=2, sorted=True)
    top2 = -values
    d_1 = top2[:, 0]
    d_2 = top2[:, 1]

    row_idx = tf.range(tf.shape(labels)[0], dtype=tf.int32)
    idx = tf.stack([row_idx, labels], axis=1)
    d_y = tf.gather_nd(dist, idx, name='dy')

    indicator = tf.cast(tf.nn.in_top_k(-dist, labels, k=1), tf.float32)
    d_c = indicator*d_2 + (1-indicator)*d_1

    loss = tf.nn.relu((d_y-d_c)/(d_y+d_c)+margin, name='loss')
    mean_loss = tf.reduce_mean(loss, name='mean_loss')

    return mean_loss

# minimum classification error loss (MCE)
def mce_loss(features, labels, centers, epsilon):
    dist = distance(features, centers)

    values, indexes = tf.nn.top_k(-dist, k=2, sorted=True)
    top2 = -values
    d_1 = top2[:, 0]
    d_2 = top2[:, 1]

    row_idx = tf.range(tf.shape(labels)[0], dtype=tf.int32)
    idx = tf.stack([row_idx, labels], axis=1)
    d_y = tf.gather_nd(dist, idx, name='dy')

    indicator = tf.cast(tf.nn.in_top_k(-dist, labels, k=1), tf.float32)
    d_c = indicator*d_2 + (1-indicator)*d_1

    measure = d_y - d_c

    loss = tf.sigmoid(epsilon*measure, name='loss')
    mean_loss = tf.reduce_mean(loss, name='mean_loss')

    return mean_loss



# distance based cross entropy loss (DCE)
def dce_loss(features, labels, centers, T, flags):
    # size = features.shape[0]
    dist = distance(features, centers, flags)#50,10,5
    mask = tf.one_hot(labels, flags.num_classes) 
    mask_rep = tf.tile(tf.expand_dims(mask, axis = 2), multiples = [1,1,flags.num_protos])
    dist_not_this_class, dist_this_class = tf.dynamic_partition(dist, tf.cast(mask_rep,tf.int32), 2)
    dist_this_class = tf.reshape(dist_this_class, (-1, 1, flags.num_protos))
    dist_not_this_class = tf.reshape(dist_not_this_class, (-1, flags.num_classes - 1, flags.num_protos))

    dist_this_class_max = tf.reduce_min(dist_this_class, axis = 2)
    dist_not_this_class_min = tf.reduce_max(dist_not_this_class, axis = 2)

    dist_not_this_class_min = tf.reshape(dist_not_this_class_min, (-1,))
    dist_this_class_max = tf.reshape(dist_this_class_max, (-1,))
    # import ipdb
    # ipdb.set_trace()

    condition_indices= tf.dynamic_partition(
        tf.range(flags.num_classes * dist.shape[0]),\
         tf.cast(tf.reshape(mask, (-1,)), tf.int32), 2)
    logits = tf.dynamic_stitch(condition_indices, [dist_not_this_class_min, dist_this_class_max])
    logits = tf.reshape(logits, (-1,flags.num_classes))
    logits = -logits / T
    mean_loss = softmax_loss(logits, labels)


    # # for another loss calculation. 
    # dist_not_this_class_max = tf.reduce_min(dist_not_this_class, axis = 2)
    # dist_not_this_class_max = tf.reshape(dist_not_this_class_max, (-1,))
    # condition_indices= tf.dynamic_partition(
    #     tf.range(flags.num_classes * dist.shape[0]),\
    #      tf.cast(tf.reshape(mask, (-1,)), tf.int32), 2)
    # logits = tf.dynamic_stitch(condition_indices, [dist_not_this_class_max, dist_this_class_max])
    # logits = tf.reshape(logits, (-1,flags.num_classes))
    # logits = -logits / T
    # mean_loss_2 = softmax_loss(logits, labels)



    return mean_loss #+ mean_loss_2

# dot product based cross entropy loss
def dot_dce_loss(features, labels, centers, T, flags):
    # size = features.shape[0]
    dist = dot_product(features, centers, flags)#50,10,5
    mask = tf.one_hot(labels, flags.num_classes) 
    mask_rep = tf.tile(tf.expand_dims(mask, axis = 2), multiples = [1,1,flags.num_protos])
    dist_not_this_class, dist_this_class = tf.dynamic_partition(dist, tf.cast(mask_rep,tf.int32), 2)
    dist_this_class = tf.reshape(dist_this_class, (-1, 1, flags.num_protos))
    dist_not_this_class = tf.reshape(dist_not_this_class, (-1, flags.num_classes - 1, flags.num_protos))

    dist_this_class_max = tf.reduce_max(dist_this_class, axis = 2)
    dist_not_this_class_min = tf.reduce_min(dist_not_this_class, axis = 2)

    dist_not_this_class_min = tf.reshape(dist_not_this_class_min, (-1,))
    dist_this_class_max = tf.reshape(dist_this_class_max, (-1,))

    condition_indices= tf.dynamic_partition(
        tf.range(flags.num_classes * dist.shape[0]),\
         tf.cast(tf.reshape(mask, (-1,)), tf.int32), 2)
    logits = tf.dynamic_stitch(condition_indices, [dist_not_this_class_min, dist_this_class_max])
    logits = tf.reshape(logits, (-1,flags.num_classes))
    logits = logits / T
    mean_loss = softmax_loss(logits, labels)
    return mean_loss


# prototype loss (PL)
def pl_loss(features, labels, centers, flags):
    batch_num = tf.cast(tf.shape(features)[0], tf.float32)
    batch_centers = tf.gather(centers, labels)
    features = tf.tile(tf.expand_dims(features, axis = 1), multiples = [1,flags.num_protos, 1])


    dis = features - batch_centers
    return tf.div(tf.nn.l2_loss(dis), batch_num)
    
##################################################
# return the training operation to train the network
def training(loss, learning_rate):
    # optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate)
    # RMSPropOptimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)
    return train_op
    
##################################################
# evaluation operation in traditional softmax-layer based NNs
def base_evaluation(logits, labels):
    prediction = tf.argmax(logits, axis=1, name='prediction')
    correct = tf.equal(tf.cast(prediction, tf.int32), labels, name='correct')
    return tf.reduce_sum(tf.cast(correct, tf.float32), name='evaluation')

# prediction operation in CPL or GCPL framwork
def predict(features, centers):
    dist = distance(features, centers) 
    prediction = tf.argmin(dist, axis=1, name='prediction')
    return tf.cast(prediction, tf.int32)

# evaluation operation in CPL or GCPL framework
def evaluation(features, labels, centers, flags):
    dist = distance(features, centers, flags)
    # mask = tf.one_hot(labels, flags.num_classes) 
    # mask_rep = tf.tile(tf.expand_dims(mask, axis = 2), multiples = [1,1,flags.num_protos])
    # dist_not_this_class, dist_this_class = tf.dynamic_partition(dist, tf.cast(mask_rep,tf.int32), 2)
    # dist_this_class = tf.reshape(dist_this_class, (-1, 1, flags.num_protos))
    # dist_not_this_class = tf.reshape(dist_not_this_class, (-1, flags.num_classes - 1, flags.num_protos))
    # dist_this_class_max = tf.reduce_min(dist_this_class, axis = 2)
    # dist_not_this_class_min = tf.reduce_max(dist_not_this_class, axis = 2)
    # dist_not_this_class_min = tf.reshape(dist_not_this_class_min, (-1,))
    # dist_this_class_max = tf.reshape(dist_this_class_max, (-1,))
    # condition_indices= tf.dynamic_partition(
    #     tf.range(flags.num_classes * dist.shape[0]),\
    #      tf.cast(tf.reshape(mask, (-1,)), tf.int32), 2)
    # logits = tf.dynamic_stitch(condition_indices, [dist_not_this_class_min, dist_this_class_max])
    # logits = tf.reshape(logits, (-1,flags.num_classes))
    # prediction = tf.argmin(logits, axis=1, name='prediction')
    size = tf.shape(features)[0]
    dist = tf.reshape(dist, [size, -1])
    prediction = tf.argmin(dist, axis = 1) // flags.num_protos
    correct = tf.equal(tf.cast(prediction, tf.int32), labels, name='correct')


    return tf.reduce_sum(tf.cast(correct, tf.float32), name='evaluation')


# evaluation operation in CPL or GCPL framework
def evaluation_dot_product(features, labels, centers, flags):
    dist = dot_product(features, centers, flags)
    # mask = tf.one_hot(labels, flags.num_classes) 
    # mask_rep = tf.tile(tf.expand_dims(mask, axis = 2), multiples = [1,1,flags.num_protos])
    # dist_not_this_class, dist_this_class = tf.dynamic_partition(dist, tf.cast(mask_rep,tf.int32), 2)
    # dist_this_class = tf.reshape(dist_this_class, (-1, 1, flags.num_protos))
    # dist_not_this_class = tf.reshape(dist_not_this_class, (-1, flags.num_classes - 1, flags.num_protos))
    # dist_this_class_max = tf.reduce_max(dist_this_class, axis = 2)
    # dist_not_this_class_min = tf.reduce_min(dist_not_this_class, axis = 2)
    # dist_not_this_class_min = tf.reshape(dist_not_this_class_min, (-1,))
    # dist_this_class_max = tf.reshape(dist_this_class_max, (-1,))
    # condition_indices= tf.dynamic_partition(
    #     tf.range(flags.num_classes * dist.shape[0]),\
    #      tf.cast(tf.reshape(mask, (-1,)), tf.int32), 2)
    # logits = tf.dynamic_stitch(condition_indices, [dist_not_this_class_min, dist_this_class_max])
    # logits = tf.reshape(logits, (-1,flags.num_classes))
    size = tf.shape(features)[0]
    dist = tf.reshape(dist, [size, -1])
    prediction = tf.argmax(dist, axis = 1) // flags.num_protos
    correct = tf.equal(tf.cast(prediction, tf.int32), labels, name='correct')


    # prediction = tf.argmax(logits, axis=1, name='prediction')
    # correct = tf.equal(tf.cast(prediction, tf.int32), labels, name='correct')
    return tf.reduce_sum(tf.cast(correct, tf.float32), name='evaluation')

def evaluation_softmax(logits, labels):
    prediction = tf.argmax(logits, axis=1, name='prediction')
    correct = tf.equal(tf.cast(prediction, tf.int32), labels, name='correct')
    return tf.reduce_sum(tf.cast(correct, tf.float32), name='evaluation')

##################################################
# construct prototypes (centers) for each class
def construct_center(features, num_classes, class_n, flags):
    len_features = features.get_shape()[1]
    centers = tf.get_variable('centers'+str(class_n), [flags.num_protos, len_features], dtype=tf.float32,\
        # initializer=tf.constant_initializer(0.1*class_n))
        initializer=tf.constant_initializer(0))
    return centers

# operations used to initialize the prototypes in
# each class (with the mean vector of the class)
def init_centers(features, labels, centers, counts):
    add_op = tf.scatter_add(centers, labels, features, name='add_op')
    unique_label, unique_index, unique_count = tf.unique_with_counts(labels)
    count_op = tf.scatter_add(counts, unique_label, unique_count, name='count_op')
    average_op = tf.assign(centers, centers/tf.cast(tf.reshape(counts, [-1,1]), tf.float32),
        name='average_op')

    return add_op, count_op, average_op



