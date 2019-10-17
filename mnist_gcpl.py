from nets import *
import functions as func
import numpy as np
import tensorflow as tf
import argparse
import time
import os
from utils import *
from tf_func import *
from densenet import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# from resnet import *

from tensorflow.examples.tutorials.mnist import input_data 
mnist = input_data.read_data_sets("./MNIST_data", one_hot=False)
FLAGS = None

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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
# def do_eval(sess, eval_correct, images, labels, test_x, test_y, logits):
#     true_count = 0.0
#     test_num = test_y.shape[0]
#     batch_size = FLAGS.batch_size
#     batch_num = test_num // batch_size if test_num % batch_size == 0 else test_num // batch_size + 1
    
#     for i in range(batch_num):
#         batch_x = test_x[i*batch_size:(i+1)*batch_size]
#         batch_y = test_y[i*batch_size:(i+1)*batch_size]
#         true_count += sess.run(eval_correct, feed_dict={images:batch_x, labels:batch_y})
#         if i == 0:
#            logits_all = np.asarray(sess.run(logits, feed_dict={images:batch_x, labels:batch_y}))
#         else:
#            logits_all = np.concatenate((logits_all, np.asarray(sess.run(logits, feed_dict={images:batch_x, labels:batch_y}))), axis = 0)
    
#     return true_count / test_num, logits_all

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
        # data normalization.
        # test_x = test_x - np.mean(test_x, axis = 0)
        # test_x = test_x / (np.std(test_x, axis = 0) + 1e-8)
        # train_x = train_x - np.mean(train_x, axis = 0)
        # train_x= train_x / (np.std(train_x, axis = 0) + 1e-8)
        train_num = train_x.shape[0]
        test_num = test_x.shape[0]
        # construct the computation graph
        images_new = tf.placeholder(tf.float32, shape=[None,1,28,28])
        images = tf.placeholder(tf.float32, shape=[None,1,28,28])
        features, logits = mnist_net(images)

        if FLAGS.use_augmentation:
            augment = data_augmentor(images_new)

        labels = tf.placeholder(tf.int32, shape=[None])
        lr= tf.placeholder(tf.float32)
        # print('test!')

    elif FLAGS.dataset == 'cifar10':
        from keras.datasets import cifar10
        (xtrain, ytrain), (xtest, ytest) = cifar10.load_data()
        ytrain = np.squeeze(ytrain)
        ytest = np.squeeze(ytest)
        xtrain = xtrain.astype('float32')
        xtrain = xtrain / 255.0
        xtest = xtest.astype('float32')
        xtest = xtest / 255.0

        # xtrain, ytrain, xtest, ytest = load_cifar()
        train_x, train_y = xtrain.reshape(-1,32,32,3), ytrain
        test_x, test_y = xtest.reshape(-1,32,32,3), ytest
        # data normalization.
        # test_x = test_x - np.mean(test_x, axis = 0)
        # test_x = test_x / np.std(test_x, axis = 0)
        # train_x = train_x - np.mean(train_x, axis = 0)
        # train_x= train_x / np.std(train_x, axis = 0)
        #another normalization method.
        train_x = train_x - [0.491,0.482,0.447]
        train_x = train_x / [0.247,0.243,0.262]
        test_x = test_x - [0.491,0.482,0.447]
        test_x = test_x / [0.247,0.243,0.262]        

        train_num = train_x.shape[0]
        test_num = test_x.shape[0]
	    # construct the computation graph
        images = tf.placeholder(tf.float32, shape=[None,32,32,3])
        # images_normalized = tf.map_fn(lambda img: tf.image.per_image_standardization(img),
        #                        images)
        images_new = tf.placeholder(tf.float32, shape=[None,32,32,3])
        labels = tf.placeholder(tf.int32, shape=[None])
        lr= tf.placeholder(tf.float32)

        if FLAGS.model == 'resnet':
            if FLAGS.use_augmentation:
                augment = data_augmentor(images_new)
            features, logits = inference(images, FLAGS.num_residual_blocks, reuse=False, weight_decay = FLAGS.weight_decay)
        else:
            if FLAGS.use_augmentation:
                augment = data_augmentor(images_new)
            features, logits = cifar_net1(images, [0.1,0.5,0.5])	



    elif FLAGS.dataset == 'cifar100':
        from keras.datasets import cifar100
        (xtrain, ytrain), (xtest, ytest) = cifar100.load_data()
        ytrain = np.squeeze(ytrain)
        ytest = np.squeeze(ytest)
        xtrain = xtrain.astype('float32')
        xtrain = xtrain / 255.0
        xtest = xtest.astype('float32')
        xtest = xtest / 255.0
        # xtrain, ytrain, xtest, ytest = load_cifar100()
        train_x, train_y = xtrain.reshape(-1,32,32, 3), ytrain
        test_x, test_y = xtest.reshape(-1,32,32,3), ytest
        #another normalization method.
        train_x = train_x - [0.4914, 0.4822, 0.4465]
        train_x = train_x / [0.2023, 0.1994, 0.2010]
        test_x = test_x - [0.4914, 0.4822, 0.4465]
        test_x = test_x / [0.2023, 0.1994, 0.2010]    

        train_num = train_x.shape[0]
        test_num = test_x.shape[0]

	    # construct the computation graph
        images = tf.placeholder(tf.float32, shape=[None,32,32,3])
        images_new = tf.placeholder(tf.float32, shape=[None,32,32,3])
        labels = tf.placeholder(tf.int32, shape=[None])
        lr= tf.placeholder(tf.float32)

        if FLAGS.model == 'densenet':
            if FLAGS.use_augmentation:
                augment = data_augmentor(images_new)
            features, logits = densenet_bc(images, num_classes = FLAGS.num_classes,is_training = True, growth_rate = 12,drop_rate = 0,\
            	depth = 100, for_imagenet = False, reuse = False, scope='test')
            # inference(images, FLAGS.num_residual_blocks, reuse=False, weight_decay = FLAGS.weight_decay)
        elif FLAGS.model == 'resnet':
            if FLAGS.use_augmentation:
                augment = data_augmentor(images_new)
            features, logits = inference(images, FLAGS.num_residual_blocks, FLAGS.num_classes,\
            	reuse=False, weight_decay = FLAGS.weight_decay)
        else:
            if FLAGS.use_augmentation:
                augment = data_augmentor(images_new)
            features, logits = cifar_net1(images, [0.1,0.5,0.5])	

    # import ipdb
    # ipdb.set_trace()
    if FLAGS.loss == 'cpl': 
        centers = []
        for i in range(FLAGS.num_classes):
            centers.append(func.construct_center(features, FLAGS.num_classes, i, FLAGS))
        centers = tf.stack(centers, 0)

        if FLAGS.use_dot_product:
            loss1 = func.dot_dce_loss(features, labels, centers, FLAGS.temp, FLAGS)
            eval_correct = func.evaluation_dot_product(features, labels, centers, FLAGS)
        else:
            loss1 = func.dce_loss(features, labels, centers, FLAGS.temp, FLAGS)
            eval_correct = func.evaluation(features, labels, centers, FLAGS)

        loss2 = FLAGS.weight_pl * func.pl_loss(features, labels, centers, FLAGS)
        loss = loss1 + loss2
        
        if FLAGS.model == 'resnet':
            # reg_losses = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss = tf.add_n([loss] + reg_losses)#loss + reg_losses
        if FLAGS.model == 'densenet':
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss = tf.add_n([loss] + reg_losses)#loss + reg_losses

        train_op = func.training(loss, FLAGS, lr)

    elif FLAGS.loss == "softmax":
        loss = func.softmax_loss(logits, labels)
        eval_correct = func.evaluation_softmax(logits, labels)
        if FLAGS.model == 'resnet':
            # reg_losses = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            # loss = loss + reg_losses
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss = tf.add_n([loss] + reg_losses)#loss + reg_losses
        if FLAGS.model == 'densenet':
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss = tf.add_n([loss] + reg_losses)#loss + reg_losses

        train_op = func.training(loss, FLAGS, lr)

    init = tf.global_variables_initializer()
    # initialize the variables
    sess = tf.Session()
    sess.run(init)
    saver = tf.train.Saver(max_to_keep=1)
    if FLAGS.restore:
        saver.restore(sess,FLAGS.restore)
        print('restore the model from: ', FLAGS.restore)

    # run the computation graph (train and test process)
    stopping = 0
    index = range(train_num)
    np.random.shuffle(list(index))
    batch_size = FLAGS.batch_size
    batch_num = train_num//batch_size if train_num % batch_size==0 else train_num//batch_size+1
    

    # train the framework with the training data
    acc_save = []
    steps = 0
    for epoch in range(FLAGS.num_epoches):
        time1 = time.time()
        loss_now = 0.0
        score_now = 0.0
        loss_dce = 0.0
        loss_pl = 0.0
        reg_loss = 0.0

        for i in range(batch_num):

            # if loss_now > loss_before or score_now < score_before:
            # if (epoch + 1) % FLAGS.decay_step == 0:
            if epoch < 10:
                # learning_rate_temp = FLAGS.learning_rate * float(steps) / float(batch_num * 10)
                learning_rate_temp = FLAGS.learning_rate 

            batch_x = train_x[index[i*batch_size:(i+1)*batch_size]]
            batch_y = train_y[index[i*batch_size:(i+1)*batch_size]]
            # mask_temp = compute_mask(FLAGS, batch_y)

            if FLAGS.loss == 'cpl': 
                if FLAGS.model == 'resnet' or FLAGS.model == 'densenet':
                    if FLAGS.use_augmentation:
                        batch_x = augment.output(sess, batch_x)
                    result = sess.run([train_op, loss, loss1, loss2, eval_correct, reg_losses, centers, features],\
                    feed_dict={images:batch_x, labels:batch_y, lr:learning_rate_temp})
                    reg_loss += np.sum(result[5])

                else:
                    if FLAGS.use_augmentation:
                        print('!')
                        batch_x = augment.output(sess, batch_x)
                    result = sess.run([train_op, loss, loss1, loss2, eval_correct, centers, features],\
                    feed_dict={images:batch_x, labels:batch_y, lr:learning_rate_temp})

                loss_now += result[1]
                score_now += result[4]
                loss_dce += result[2]
                loss_pl += result[3]

                if i == 0:
                    features_container = np.asarray(result[-1])
                    label_container = batch_y
                else:
                    features_container = np.concatenate((features_container, result[-1]), axis=0)
                    label_container = np.concatenate((label_container, batch_y), axis=0)


            elif FLAGS.loss == 'softmax':
                if FLAGS.model == 'resnet' or FLAGS.model == 'densenet':
                    if FLAGS.use_augmentation:
                        batch_x = augment.output(sess, batch_x)

                    result = sess.run([train_op, loss, eval_correct, reg_losses],\
                    feed_dict={images:batch_x, labels:batch_y, lr:learning_rate_temp})

                    reg_loss += np.sum(result[3])
                else:
                    if FLAGS.use_augmentation:
                        batch_x = augment.output(sess, batch_x)

                    result = sess.run([train_op, loss, eval_correct],\
                    feed_dict={images:batch_x, labels:batch_y, lr:learning_rate_temp})

                loss_now += result[1]
                score_now += result[2]
            steps += 1

        # for visualization. 
        if FLAGS.loss == 'cpl' and FLAGS.dataset == 'mnist':
            if epoch % 10 == 0:
                centers_container = result[-2]
                func.visualize(features_container, label_container, epoch, centers_container, FLAGS)

        # if epoch +1 == 150 or epoch +1 == 225:
        if (epoch + 1) % FLAGS.decay_step == 0:
            stopping += 1
            learning_rate_temp *= FLAGS.decay_rate
            print ("\033[1;31;40mdecay learning rate {}th time!\033[0m".format(stopping))


        score_now /= train_num
        loss_now /= batch_num

        if FLAGS.loss == 'cpl':
            loss_dce = loss_dce / batch_num
            loss_pl = loss_pl / batch_num
            if FLAGS.model == 'resnet' or FLAGS.model == 'densenet':
                reg_loss /= batch_num
                print ('epoch {}: training: loss --> {:.3f}, dce_loss --> {:.3f}, pl_loss --> {:.3f}, reg_loss --> {:.3f},\
                 acc --> {:.3f}%'.format(epoch, loss_now, loss_dce, loss_pl, reg_loss, score_now*100))
            else:
                print ('epoch {}: training: loss --> {:.3f}, dce_loss --> {:.3f}, pl_loss --> {:.3f},\
                 acc --> {:.3f}%'.format(epoch, loss_now, loss_dce, loss_pl, score_now*100))
        elif FLAGS.loss == 'softmax':
            if FLAGS.model == 'resnet' or FLAGS.model == 'densenet':
                reg_loss /= batch_num
                print ('epoch {}: training: loss --> {:.3f}, reg_loss --> {:.3f},\
                   acc --> {:.3f}%'.format(epoch, loss_now, reg_loss, score_now*100)) 
            else:
                print ('epoch {}: training: loss --> {:.3f},\
                   acc --> {:.3f}%'.format(epoch, loss_now, score_now*100))        	


        # epoch += 1
        np.random.shuffle(list(index))
        time2 = time.time()
        print ('time for this epoch: {:.3f} minutes'.format((time2-time1)/60.0))


        # test the framework with the test data
        if (epoch + 1) % FLAGS.print_step == 0:
            # test_score, logits_test = do_eval(sess, eval_correct, images, labels, test_x, test_y, logits)
            # np.save('./cifar10_logits.npy', logits_test)
            test_score = do_eval(sess, eval_correct, images, labels, test_x, test_y)
            print ('epoch:{}, accuracy on the test dataset: {:.3f}%'.format(epoch, test_score*100))
            acc_save.append(test_score)
            temp = np.amax(np.asarray(acc_save))
            print('best test acc:',temp)

        # saving the model.
        if not os.path.isdir(os.path.join(FLAGS.log_dir, FLAGS.dataset)):  
            os.makedirs(os.path.join(FLAGS.log_dir, FLAGS.dataset))
        checkpoint_file = os.path.join(str(os.path.join(FLAGS.log_dir, FLAGS.dataset)),\
         'model_'+FLAGS.loss+'_'+str(FLAGS.use_dot_product)+'_'+str(FLAGS.learning_rate)+'_'+str(FLAGS.batch_size)+'.ckpt')
        saver.save(sess, checkpoint_file, global_step=epoch)



    acc_save = np.asarray(acc_save)
    np.save('./acc_test_cifar10_original_paper.npy', acc_save)
    sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for training')
    parser.add_argument('--weight_pl', type=float, default=0.0001, help='the weight for the prototype loss (PL)')
    parser.add_argument('--num_epoches', type=int, default=300, help='the number of the epoches')
    parser.add_argument('--use_dot_product', type=str2bool, default=True, help='what metric we use in cpl loss.')
    parser.add_argument('--weight_decay', type=float, default=0.0002, help='weight decay for resnet model.')
    parser.add_argument('--optimizer', type=str, default='MOM', help='optimizer for the model.',\
    	choices=['ADAGRAD', 'ADAM',  'MOM','SGD', 'RMSP'])
    parser.add_argument('--model', type=str, default='resnet', help='which model to use for training.')
    parser.add_argument('--use_augmentation', type=str2bool, default=False, help = 'whether to use data augmentation during training.')
    parser.add_argument('--decay_rate', type=float, default=0.5, help='decay rate.')
    parser.add_argument('--decay_step', type=float, default=60, help='the steps to decay the learning rate')
    parser.add_argument('--num_classes', type=int, default=100, help='the number of the classes')

    parser.add_argument('--dataset', type=str, default='mnist', help='which kind of data we use')
    parser.add_argument('--stop', type=int, default=np.inf, help='stopping number')
    parser.add_argument('--temp', type=float, default=1.0, help='the temperature used for calculating the loss')
    parser.add_argument('--gpu', type=int, default=1, help='the gpu id for use')    
    parser.add_argument('--num_protos', type=int, default=5, help='the number of the protos')
    parser.add_argument('--print_step', type=int, default=1, help='the number steps for printing.')
    parser.add_argument('--loss', type=str, default='cpl', help='which loss to choose.')
    parser.add_argument('--num_residual_blocks', type=int, default=3, help='the number of residual blocks in the resnet.')#Resnet 6n+2
    parser.add_argument('--log_dir', type=str, default='./model', help='where to save the model.')
    parser.add_argument('--restore', type=str, default = '', help = 'whether to restore model.')

    

    FLAGS = parser.parse_args()
    print (150*'*')
    print ('Configuration of the training:')
    print ('learning rate:', FLAGS.learning_rate)
    print ('batch size:', FLAGS.batch_size)
    print ('stopping:', FLAGS.stop)
    print ('learning rate decay step:', FLAGS.decay_step)
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
    print ('if use dot product:', FLAGS.use_dot_product)
    print ('weight decay rate is: ', FLAGS.weight_decay)
    print ('use data augmentation: ', FLAGS.use_augmentation)
    print ('number of epoches:', FLAGS.num_epoches)
    print ('whether to restore model:', FLAGS.restore)
    print ('which optimizer used in the model:', FLAGS.optimizer)



    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    run_training()

