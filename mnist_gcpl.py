from nets import *
import functions as func
import numpy as np
import tensorflow as tf
import argparse
import time
import os
from utils import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.examples.tutorials.mnist import input_data 
mnist = input_data.read_data_sets("./MNIST_data", one_hot=False)
FLAGS = None

# compute accuracy on the test dataset
def do_eval(sess, eval_correct, images, labels, test_x, test_y):
    true_count = 0.0
    test_num = test_y.shape[0]
    batch_size = FLAGS.batch_size
    batch_num = test_num // batch_size if test_num % batch_size == 0 else test_num // batch_size + 1

    for i in range(batch_num):
        batch_x = test_x[i*batch_size:(i+1)*batch_size]
        batch_y = test_y[i*batch_size:(i+1)*batch_size]
        true_count += sess.run(eval_correct, feed_dict={images:batch_x, labels:batch_y})
    
    return true_count / test_num

# initialize the prototype with the mean vector (on the train dataset) of the corresponding class
def compute_centers(sess, add_op, count_op, average_op, images_placeholder, labels_placeholder, train_x, train_y):
    train_num = train_y.shape[0]
    batch_size = FLAGS.batch_size
    batch_num = train_num // batch_size if train_num % batch_size == 0 else train_num // batch_size + 1

    for i in range(batch_num):
        batch_x = train_x[i*batch_size:(i+1)*batch_size]
        batch_y = train_y[i*batch_size:(i+1)*batch_size]
        sess.run([add_op, count_op], feed_dict={images_placeholder:batch_x, labels_placeholder:batch_y})

    sess.run(average_op)


def run_training():

    # load the data
    print (150*'*')
    if FLAGS.dataset == 'mnist':
        train_x, train_y = mnist.train.images.reshape(-1,1,28,28), mnist.train.labels
        test_x, test_y = mnist.test.images.reshape(-1,1,28,28), mnist.test.labels
        train_num = train_x.shape[0]
        test_num = test_x.shape[0]
        # construct the computation graph
        images = tf.placeholder(tf.float32, shape=[None,1,28,28])
        labels = tf.placeholder(tf.int32, shape=[None])
        lr= tf.placeholder(tf.float32)

        features, logits = mnist_net(images)

    elif FLAGS.dataset == 'cifar10':
        xtrain, ytrain, xtest, ytest = load_cifar()
        train_x, train_y = xtrain.reshape(-1,32,32, 3), ytrain
        test_x, test_y = xtest.reshape(-1,32,32,3), ytest
        train_num = train_x.shape[0]
        test_num = test_x.shape[0]

	    # construct the computation graph
        images = tf.placeholder(tf.float32, shape=[None,32,32, 3])
        labels = tf.placeholder(tf.int32, shape=[None])
        lr= tf.placeholder(tf.float32)

        if FLAGS.model == 'resnet':
            features, logits = inference(images, FLAGS.num_residual_blocks, reuse=False)
        else:
        	features, logits = cifar_net1(images, [0.1,0.5,0.5])		



    if FLAGS.loss == 'cpl': 
        centers = []
        for i in range(FLAGS.num_classes):
            centers.append(func.construct_center(features, FLAGS.num_classes, i, FLAGS))
        centers = tf.stack(centers, 0)
        loss1 = func.dce_loss(features, labels, centers, FLAGS.temp, FLAGS)
        loss2 = func.pl_loss(features, labels, centers, FLAGS)
        loss = loss1 + FLAGS.weight_pl * loss2
        eval_correct = func.evaluation(features, labels, centers, FLAGS)
        train_op = func.training(loss, lr)

        if FLAGS.model == 'resnet':
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss += reg_losses 

    elif FLAGS.loss == "softmax":
        loss = func.softmax_loss(logits, labels)
        eval_correct = func.evaluation_softmax(logits, labels)
        train_op = func.training(loss, lr)
        if FLAGS.model == 'resnet':
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss += reg_losses 
     
    #counts = tf.get_variable('counts', [FLAGS.num_classes], dtype=tf.int32,
    #    initializer=tf.constant_initializer(0), trainable=False)
    #add_op, count_op, average_op = net.init_centers(features, labels, centers, counts)
    
    init = tf.global_variables_initializer()

    # initialize the variables
    sess = tf.Session()
    sess.run(init)
    #compute_centers(sess, add_op, count_op, average_op, images, labels, train_x, train_y)

    # run the computation graph (train and test process)
    epoch = 1
    loss_before = np.inf
    score_before = 0.0
    stopping = 0
    index = range(train_num)
    np.random.shuffle(list(index))
    batch_size = FLAGS.batch_size
    batch_num = train_num//batch_size if train_num % batch_size==0 else train_num//batch_size+1
    #saver = tf.train.Saver(max_to_keep=1)

    # train the framework with the training data
    while stopping<FLAGS.stop:
        time1 = time.time()
        loss_now = 0.0
        score_now = 0.0
        loss_dce = 0.0
        loss_pl = 0.0
        reg_loss = 0.0

    
        for i in range(batch_num):
            batch_x = train_x[index[i*batch_size:(i+1)*batch_size]]
            batch_y = train_y[index[i*batch_size:(i+1)*batch_size]]
            # mask_temp = compute_mask(FLAGS, batch_y)

            if FLAGS.loss == 'cpl': 
                if FLAGS.model == 'resnet':
                    result = sess.run([train_op, loss, loss1, loss2, eval_correct, reg_losses],\
                    feed_dict={images:batch_x, labels:batch_y, lr:FLAGS.learning_rate})
                    reg_loss += result[5]
                else:
                    result = sess.run([train_op, loss, loss1, loss2, eval_correct],\
                    feed_dict={images:batch_x, labels:batch_y, lr:FLAGS.learning_rate})

                loss_now += result[1]
                score_now += result[4]
                loss_dce += result[2]
                loss_pl += result[3]
            elif FLAGS.loss == 'softmax':
                if FLAGS.model == 'resnet':
                    result = sess.run([train_op, loss, eval_correct, reg_losses],\
                    feed_dict={images:batch_x, labels:batch_y, lr:FLAGS.learning_rate})
                    reg_loss += result[3]
                else:
                    result = sess.run([train_op, loss, eval_correct],\
                    feed_dict={images:batch_x, labels:batch_y, lr:FLAGS.learning_rate})

                loss_now = result[1]
                score_now += result[2]

        score_now /= train_num

        if FLAGS.loss == 'cpl':
            if FLAGS.model == 'resnet':
               print ('epoch {}: training: loss --> {:.3f}, dce_loss --> {:.3f}, pl_loss --> {:.3f}, reg_loss --> {:.3f},\
                 acc --> {:.3f}%'.format(epoch, loss_now, loss_dce, loss_pl, reg_loss, score_now*100))
            else:
               print ('epoch {}: training: loss --> {:.3f}, dce_loss --> {:.3f}, pl_loss --> {:.3f},\
                 acc --> {:.3f}%'.format(epoch, loss_now, loss_dce, loss_pl, score_now*100))
        elif FLAGS.loss == 'softmax':
            if FLAGS.model == 'resnet':
                print ('epoch {}: training: loss --> {:.3f}, reg_loss --> {:.3f},\
                   acc --> {:.3f}%'.format(epoch, loss_now, reg_loss, score_now*100)) 
            else:
                print ('epoch {}: training: loss --> {:.3f},\
                   acc --> {:.3f}%'.format(epoch, loss_now, score_now*100))        	
        #print sess.run(centers)
    
        if loss_now > loss_before or score_now < score_before:
            stopping += 1
            FLAGS.learning_rate *= FLAGS.decay
            print ("\033[1;31;40mdecay learning rate {}th time!\033[0m".format(stopping))
            
        loss_before = loss_now
        score_before = score_now

        #checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
        #saver.save(sess, checkpoint_file, global_step=epoch)

        epoch += 1
        np.random.shuffle(list(index))
        time2 = time.time()
        print ('time for this epoch: {:.3f} minutes'.format((time2-time1)/60.0))


        # test the framework with the test data
        if epoch % FLAGS.print_step == 0:
            test_score = do_eval(sess, eval_correct, images, labels, test_x, test_y)
            print ('epoch:{}, accuracy on the test dataset: {:.3f}%'.format(epoch, test_score*100))

        # epoch += 1
        


    sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size for training')
    parser.add_argument('--dataset', type=str, default='cifar10', help='which kind of data we use')
    parser.add_argument('--stop', type=int, default=100, help='stopping number')
    parser.add_argument('--decay', type=float, default=0.9, help='the value to decay the learning rate')
    parser.add_argument('--temp', type=float, default=1.0, help='the temperature used for calculating the loss')
    parser.add_argument('--weight_pl', type=float, default=0.001, help='the weight for the prototype loss (PL)')
    parser.add_argument('--gpu', type=int, default=0, help='the gpu id for use')
    parser.add_argument('--num_classes', type=int, default=10, help='the number of the classes')
    parser.add_argument('--num_protos', type=int, default=5, help='the number of the protos')
    parser.add_argument('--print_step', type=int, default=10, help='the number steps for printing.')
    parser.add_argument('--loss', type=str, default='cpl', help='which loss to choose.')
    parser.add_argument('--model', type=str, default='resnet', help='which model to use for training.')
    parser.add_argument('--num_residual_blocks', type=int, default=5, help='the number of residual blocks in the resnet.')

    

    FLAGS, unparsed = parser.parse_known_args()
    print (150*'*')
    print ('Configuration of the training:')
    print ('learning rate:', FLAGS.learning_rate)
    print ('batch size:', FLAGS.batch_size)
    print ('stopping:', FLAGS.stop)
    print ('learning rate decay:', FLAGS.decay)
    print ('value of the temperature:', FLAGS.temp)
    print ('prototype loss weight:', FLAGS.weight_pl)
    print ('number of classes:', FLAGS.num_classes)
    print ('GPU id:', FLAGS.gpu)
    print ('Data used:', FLAGS.dataset)
    print ('number of protos:', FLAGS.num_protos )
    print ('printing steps:', FLAGS.print_step)
    print ('loss function:', FLAGS.loss)
    print ('Model type:', FLAGS.model)
    print ('number of residual blocks:', FLAGS.num_residual_blocks)


    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

    run_training()

