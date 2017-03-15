# -*- coding: utf-8 -*-
# Created by liangzh0707 on 2017/3/2
import cv2
import numpy as np
import os
import tensorflow as tf
import cPickle


D = {"NE": 0, "DI": 1, "FE": 2, "HP": 3, "AN": 4, "SA": 5, "SU": 6}
status = ["自然状态", "恶心", "害怕", "快乐", "生气", "难过", "惊讶"]


def weight_variable(shape, name, wd=0.0, stddev=0.1):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name)
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def bias_variable(shape, name, constant=0.0):
    return tf.Variable(tf.constant(constant, shape=shape), name)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def op_max_pool_2x2(x, use=False):
    if use:
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    else:
        return x


def op_normalize(x, use=False, name='unnamed'):
    if use:
        return tf.nn.lrn(x, 5, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)
    else:
        return x


def op_dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)


def inference(x):
    x_image = tf.reshape(x, [-1, 48, 48, 1])
    keep_prob = tf.placeholder(tf.float32)
    USE_POOL = False
    USE_NORM = True

    with tf.variable_scope('conv1') as scope:
        kernel = weight_variable([3, 3, 1, 32], "weights")
        bias = bias_variable([32], "biases")
        conv = conv2d(x_image, kernel)
        pre_activation = tf.nn.bias_add(conv, bias)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    norm1 = op_normalize(conv1, USE_NORM, 'norm1')
    pool1 = op_max_pool_2x2(norm1, USE_POOL)
    drop1 = op_dropout(pool1, keep_prob)

    with tf.variable_scope('conv2') as scope:
        kernel = weight_variable([3, 3, 32, 32], "weights")
        bias = bias_variable([32], "biases")
        conv = conv2d(drop1, kernel)
        pre_activation = tf.nn.bias_add(conv, bias)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)

    norm2 = op_normalize(conv2, USE_NORM, name='norm2')
    pool2 = op_max_pool_2x2(norm2, USE_POOL)
    drop2 = op_dropout(pool2, keep_prob)

    with tf.variable_scope('conv3') as scope:
        kernel = weight_variable([3, 3, 32, 32], "weights")
        bias = bias_variable([32], "biases")
        conv = conv2d(drop2, kernel)
        pre_activation = tf.nn.bias_add(conv, bias)
        conv3 = tf.nn.relu(pre_activation, name=scope.name)

    norm3 = op_normalize(conv3, USE_NORM, name='norm3')
    pool3 = op_max_pool_2x2(norm3, USE_POOL)
    drop3 = op_dropout(pool3, keep_prob)

    USE_POOL = True

    with tf.variable_scope('conv4') as scope:
        kernel = weight_variable([3, 3, 32, 32], "weights")
        bias = bias_variable([32], "biases")
        conv = conv2d(drop3, kernel)
        pre_activation = tf.nn.bias_add(conv, bias)
        conv4 = tf.nn.relu(pre_activation, name=scope.name)

    norm4 = op_normalize(conv4, USE_NORM, name='norm4')
    pool4 = op_max_pool_2x2(norm4, USE_POOL)
    drop4 = op_dropout(pool4, keep_prob)

    with tf.variable_scope('conv5') as scope:
        kernel = weight_variable([3, 3, 32, 64], "weights")
        bias = bias_variable([64], "biases")
        conv = conv2d(drop4, kernel)
        pre_activation = tf.nn.bias_add(conv, bias)
        conv5 = tf.nn.relu(pre_activation, name=scope.name)

    norm5 = op_normalize(conv5, USE_NORM, name='norm5')
    pool5 = op_max_pool_2x2(norm5, USE_POOL)
    drop5 = op_dropout(pool5, keep_prob)

    with tf.variable_scope('conv6') as scope:
        kernel = weight_variable([3, 3, 64, 128], "weights")
        bias = bias_variable([128], "biases")
        conv = conv2d(drop5, kernel)
        pre_activation = tf.nn.bias_add(conv, bias)
        conv6 = tf.nn.relu(pre_activation, name=scope.name)

    norm6 = op_normalize(conv6, USE_NORM, name='norm6')
    pool6 = op_max_pool_2x2(norm6, USE_POOL)
    # drop6 = op_dropout(pool6, keep_prob)

    with tf.variable_scope('local1') as scope:
        reshape = tf.reshape(pool6, [-1, 6 * 6 * 128])
        weights = weight_variable([6 * 6 * 128, 1024], "weights", wd=0.001)
        bias = bias_variable([1024], "biases")
        local1 = tf.nn.relu(tf.matmul(reshape, weights) + bias, name=scope.name)

    local_drop1 = op_dropout(local1, keep_prob)

    with tf.variable_scope('local2') as scope:
        weights = weight_variable([1024, 7], "weights", wd=0.001)
        bias = bias_variable([7], "biases")
        y_conv = tf.add(tf.matmul(local_drop1, weights), bias, name=scope.name)

    return y_conv, {}, {"keep_prob": keep_prob}


def loss(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    tf.add_to_collection('losses', loss)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def training(images, labels, test_images, test_labels, keys, values, images_holder, labels_holder, l, e, sess, learning_rate=1e-4, reload=False, model_file=""):

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(l)
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    if reload is True:
        saver.restore(sess, model_file)
        return

    for i in range(2000):
        mask = np.random.randint(0, labels.shape[0], (500,), np.int)

        X = images[mask]
        Y = labels[mask]
        feed_dict = {images_holder: X, labels_holder: Y, keys["keep_prob"]: values["keep_prob"]}

        _, loss_value = sess.run([train_step, l],
                                 feed_dict=feed_dict)

        if i % 10 == 0:
            saver.save(sess, "fer_input_48_48_1_out_7", global_step=i)

            feed_dict = {images_holder: test_images, labels_holder: test_labels, keys["keep_prob"]: 1.0}

            ac = sess.run(e, feed_dict=feed_dict)

            print('Step %d: loss = %.2f (%.3f )' % (i, loss_value, ac))


def evaluation(logits, labels_holder):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_holder, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, "float"))


def training_fer2013():
    with tf.Graph().as_default():
        sess = tf.InteractiveSession()
        file = open("fer2013full.cpickle", "rb")

        train_image, train_label, test_image, test_label = cPickle.load(file)

        print ("数据读取完成")
        print ("训练数据%s,测试数据%s" % (len(train_image), len(test_image)))
        assert len(train_image) > 0 and len(test_image) > 0

        images_holder = tf.placeholder(tf.float32, shape=(None, 48, 48, 1))
        labels_holder = tf.placeholder(tf.float32, shape=(None, 7))

        logits, params, mid_res = inference(images_holder)
        e = evaluation(logits, labels_holder)

        keys = {}
        values = {}
        keys["keep_prob"] = mid_res["keep_prob"]
        values["keep_prob"] = 1.0

        l = loss(logits, labels_holder)
        training(np.array(train_image), np.array(train_label), np.array(test_image), np.array(test_label),
                 keys, values, images_holder, labels_holder, l, e, sess=sess, reload=False)


if "__main__" == __name__:
    training_fer2013()
    pass

