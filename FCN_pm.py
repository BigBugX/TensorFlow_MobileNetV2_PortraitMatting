__author__ = 'Will@PCVG'
# An Implementation based on shekkizh's FCN.tensorflow
# Utils used with tensorflow implemetation
from __future__ import print_function

from __future__ import absolute_import
from __future__ import division
import copy
import functools


from ops_dup import *

import tensorflow as tf
import numpy as np

import TensorflowUtils_plus as utils
import datetime
from portrait_plus import BatchDatset, TestDataset
from PIL import Image
from scipy import misc
import os
from tensorflow.python import pywrap_tensorflow

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "1", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")
is_train = True

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'
mnv2_param_path = os.getcwd()
mnv2_model_path = os.getcwd()
model_path = os.getcwd() + "\\M2test\\mv2_1_160\\mobilenet_v2_1.0_160.ckpt"

MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 2
IMAGE_WIDTH = 600
IMAGE_HEIGHT = 800

Conv = 'MobilenetV2/Conv/weights'
expanded_conv = 'MobilenetV2/expanded_conv/project/weights'
expanded_conv_1 = 'MobilenetV2/expanded_conv_1/project/weights'
expanded_conv_2 = 'MobilenetV2/expanded_conv_2/project/weights'
expanded_conv_3 = 'MobilenetV2/expanded_conv_3/project/weights'
expanded_conv_4 = 'MobilenetV2/expanded_conv_4/project/weights'
expanded_conv_5 = 'MobilenetV2/expanded_conv_5/project/weights'
expanded_conv_6 = 'MobilenetV2/expanded_conv_6/project/weights'
expanded_conv_7 = 'MobilenetV2/expanded_conv_7/project/weights'
expanded_conv_8 = 'MobilenetV2/expanded_conv_8/project/weights'
expanded_conv_9 = 'MobilenetV2/expanded_conv_9/project/weights'
expanded_conv_10 = 'MobilenetV2/expanded_conv_10/project/weights'
expanded_conv_11 = 'MobilenetV2/expanded_conv_11/project/weights'
expanded_conv_12 = 'MobilenetV2/expanded_conv_12/project/weights'
expanded_conv_13 = 'MobilenetV2/expanded_conv_13/project/weights'
expanded_conv_14 = 'MobilenetV2/expanded_conv_14/project/weights'
expanded_conv_15 = 'MobilenetV2/expanded_conv_15/project/weights'
expanded_conv_16 = 'MobilenetV2/expanded_conv_16/project/weights'
Conv_1 = 'MobilenetV2/Conv_1/weights'
"""
layers = (
    Conv, expanded_conv,
    
    expanded_conv_1, expanded_conv_2, 
    
    expanded_conv_3, expanded_conv_4, expanded_conv_5,
    
    expanded_conv_6, expanded_conv_7, expanded_conv_8, expanded_conv_9,
    expanded_conv_10, expanded_conv_11, expanded_conv_12,
    
    expanded_conv_13, expanded_conv_14, expanded_conv_15,
    expanded_conv_16, Conv_1
)
"""

net_prefix = 'MobilenetV2/'

expanded_layer = ['expanded_conv_1', 'expanded_conv_2', 'expanded_conv_3',
                  'expanded_conv_4', 'expanded_conv_5', 'expanded_conv_6', 'expanded_conv_7',
                  'expanded_conv_8', 'expanded_conv_9', 'expanded_conv_10', 'expanded_conv_11',
                  'expanded_conv_12', 'expanded_conv_13', 'expanded_conv_14', 'expanded_conv_15',
                  'expanded_conv_16']
expanded_sub_layer = ['/expand/weights', '/depthwise/depthwise_weights', '/project/weights']

ocas = [[2,24], [1,24], [2,32], [1,32], [1,32], 
        [2,64], [1,64], [1,64], [1,64], [1,96],
        [1,96], [1,96], [2,160], [1,160], [1,160], [1, 320]] # Output_Channel And Strides for each layer



def mob_net(weights, image):

    exp = 6
    net = {}
    tmp_kernel_lst = []
    current = image

    kernels = weights['Conv']
    kernels = utils.get_variable_dup(kernels[0], name=Conv)
    current = conv2d_block(image, kernels, 32, 3, 2, is_train, name=(net_prefix+'Conv'+expanded_sub_layer[2]))  # size/2
    net['Conv'] = current

    kernels = weights['expanded_conv']
    kernels_0 = kernels[0]
    kernels_0 = utils.get_variable_dup(kernels_0, name=(net_prefix+'expanded_conv'+expanded_sub_layer[1]))
    kernels_1 = kernels[1]
    kernels_1 = utils.get_variable_dup(kernels_1, name=(net_prefix+'expanded_conv'+expanded_sub_layer[2]))
    kernels = []
    kernels.append(kernels_0)
    kernels.append(kernels_1)
    current = res_block_expt(current, kernels, 1, 16, 1, is_train, name='expanded_conv', shortcut=False)
    net['expanded_conv'] = current

    i = 0
    for layer_name in expanded_layer:
        shortcut = True
        if layer_name in ['expanded_conv_1', 'expanded_conv_3', 'expanded_conv_6',
                          'expanded_conv_10', 'expanded_conv_13']:
            shortcut = False
        kernels_0 = weights[layer_name][0]
        kernels_0 = utils.get_variable_dup(kernels_0, name=(net_prefix+layer_name+expanded_sub_layer[0]))
        kernels_1 = weights[layer_name][1]
        kernels_1 = utils.get_variable_dup(kernels_1, name=(net_prefix+layer_name+expanded_sub_layer[1]))
        kernels_2 = weights[layer_name][2]
        kernels_2 = utils.get_variable_dup(kernels_2, name=(net_prefix+layer_name+expanded_sub_layer[2]))
        kernels = []
        kernels.append(kernels_0)
        kernels.append(kernels_1)
        kernels.append(kernels_2)
        current = res_block(current, kernels, exp, ocas[i][1], ocas[i][0], is_train, name=layer_name, shortcut=shortcut)
        i += 1
        net[layer_name] = current
    
    kernels = weights['Conv_1']
    kernels = utils.get_variable_dup(kernels[0], name=(net_prefix+'Conv_1'+expanded_sub_layer[2]))
    current = conv2d_block(current, kernels, 1280, 1, 1, is_train, name='Conv_1')  # size/2
    net['Conv_1'] = current

    return net


def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    ckpt_path = model_path

    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
    vl = reader.get_variable_to_shape_map() # vl: variable(weights) list

    weights = {}
    # get each layer's veriable  
    layer_w = []
    layer_w.append(reader.get_tensor(net_prefix + 'Conv' +'/weights'))
    weights['Conv'] = layer_w
    # print(weights)


    layer_w = []
    layer_w.append(reader.get_tensor(net_prefix + 'expanded_conv' + '/depthwise/depthwise_weights'))
    layer_w.append(reader.get_tensor(net_prefix + 'expanded_conv' + '/project/weights'))
    weights['expanded_conv'] = layer_w
  

    for layer in expanded_layer:
        layer_w = []
        for sub_layer in expanded_sub_layer:
            layer_w.append(reader.get_tensor(net_prefix + layer + sub_layer))
        weights[layer] = layer_w


    layer_w = []
    layer_w.append(reader.get_tensor(net_prefix + 'Conv_1' +'/weights'))
    weights['Conv_1'] = layer_w


    print("setting up mobilenetv2 pretrained model ...")
    

    with tf.variable_scope("inference"):
        image_net = mob_net(weights, image)
        conv_final_layer = image_net['Conv_1']

        deconv_shape1 = image_net["expanded_conv_12"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, 1280], name="W_t1")
        conv_t1 = utils.conv2d_transpose_strided_expt(conv_final_layer, W_t1, output_shape=tf.shape(image_net["expanded_conv_12"]))
        fuse_1 = tf.add(conv_t1, image_net["expanded_conv_12"], name="fuse_1")

        deconv_shape2 = image_net["expanded_conv_5"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        conv_t2 = utils.conv2d_transpose_strided_expt(fuse_1, W_t2, output_shape=tf.shape(image_net["expanded_conv_5"]))
        fuse_2 = tf.add(conv_t2, image_net["expanded_conv_5"], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        conv_t3 = utils.conv2d_transpose_strided_expt(fuse_2, W_t3, output_shape=deconv_shape3, stride=8)

        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), conv_t3


def train(loss_val, var_list):
    optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 6], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name="annotation")

    pred_annotation, logits = inference(image, keep_probability)
    #tf.image_summary("input_image", image, max_images=2)
    #tf.image_summary("ground_truth", tf.cast(annotation, tf.uint8), max_images=2)
    #tf.image_summary("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_images=2)
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits,
                                                                          labels = tf.squeeze(annotation, squeeze_dims=[3]),
                                                                          name="entropy")))

    trainable_var = tf.trainable_variables()
    train_op = train(loss, trainable_var)

    train_dataset_reader = BatchDatset('data/trainlist.mat')

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()
    #summary_writer = tf.train.SummaryWriter(FLAGS.logs_dir, sess.graph)

    sess.run(tf.initialize_all_variables())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    #if FLAGS.mode == "train":
    itr = 0
    train_images, train_annotations = train_dataset_reader.next_batch()
    trloss = 0.0
    while len(train_annotations) > 0:
        #train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
        #print('==> batch data: ', train_images[0][100][100], '===', train_annotations[0][100][100])
        feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.5}
        _, rloss = sess.run([train_op, loss], feed_dict=feed_dict)
        trloss += rloss

        if itr % 10 == 0:
            #train_loss, rpred = sess.run([loss, pred_annotation], feed_dict=feed_dict)
            print("Step: %d, Train_loss:%f" % (itr, trloss / 10))
            trloss = 0.0
            #summary_writer.add_summary(summary_str, itr)

        #if itr % 10000 == 0 and itr > 0:
        '''
        valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
        valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations,
                                                       keep_probability: 1.0})
        print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))'''
        itr += 1

        train_images, train_annotations = train_dataset_reader.next_batch()
    saver.save(sess, FLAGS.logs_dir + "plus_plus_model.ckpt", itr)



def pred():
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 6], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name="annotation")

    pred_annotation, logits = inference(image, keep_probability)
    sft = tf.nn.softmax(logits)
    test_dataset_reader = TestDataset('data/testlist.mat')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        saver = tf.train.Saver()
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored...")

        for batch_count in range(0,1):
            test_images, test_annotations, test_orgs = test_dataset_reader.next_batch()        
            if len(test_annotations) > 0:
                    feed_dict = {image: test_images, annotation: test_annotations, keep_probability: 0.5}
                    preds = sess.run(pred_annotation, feed_dict=feed_dict)
                    for img_count in range (0, len(test_images)):
                        org_im = Image.fromarray(np.uint8(test_orgs[img_count]))
                        print(test_orgs[img_count].shape)
                        org_im.save('res/org%d.jpg' % batch_count)
                        save_mask_img(test_annotations[img_count], 'res/ann%d' % batch_count)
                        save_mask_img(preds[img_count], 'res/pre%d' % batch_count)

def read_org_img(path):
    image = misc.imread(path, 'RGB');
    return image

def save_mask_img(mat, name):
    w, h = mat.shape[0], mat.shape[1]
    amat = np.zeros((w, h, 3), dtype=np.float)
    amat[:,:,0:3] = mat
    misc.imsave(name + '.png', amat)
    

def save_alpha_img(org, mat, name):
    w, h = mat.shape[0], mat.shape[1]
    #print(mat[200:210, 200:210])
    rmat = np.reshape(mat, (w, h))
    amat = np.zeros((w, h, 4), dtype=np.int)
    amat[:, :, 3] = np.round(rmat * 1000)
    amat[:, :, 0:3] = org
    #print(amat[200:205, 200:205])
    #im = Image.fromarray(np.uint8(amat))
    #im.save(name + '.png')
    misc.imsave(name + '.png', amat)

if __name__ == "__main__":
    tf.app.run()
    #pred()
