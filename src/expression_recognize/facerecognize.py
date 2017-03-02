# coding:utf-8
import cv2
import numpy as np
import tensorflow as tf

import facedetection as fapi
from src.data_collector import data_input

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


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def inference(x):
    x_image = tf.reshape(x, [-1, 80, 80, 1])
    keep_prob = tf.placeholder(tf.float32)

    with tf.variable_scope('conv1') as scope:
        kernel = weight_variable([5, 5, 1, 32], "weights")
        bias = bias_variable([32], "biases")
        conv = conv2d(x_image, kernel)
        pre_activation = tf.nn.bias_add(conv, bias)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    norm1 = tf.nn.lrn(conv1, 5, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    pool1 = max_pool_2x2(norm1)
    drop1 = tf.nn.dropout(pool1, keep_prob)

    with tf.variable_scope('conv2') as scope:
        kernel = weight_variable([5, 5, 32, 64], "weights")
        bias = bias_variable([64], "biases")
        conv = conv2d(drop1, kernel)
        pre_activation = tf.nn.bias_add(conv, bias)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)

    norm2 = tf.nn.lrn(conv2, 5, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    pool2 = max_pool_2x2(conv2)
    drop2 = tf.nn.dropout(pool2, keep_prob)

    with tf.variable_scope('conv3') as scope:
        kernel = weight_variable([5, 5, 64, 128], "weights")
        bias = bias_variable([128], "biases")
        conv = conv2d(drop2, kernel)
        pre_activation = tf.nn.bias_add(conv, bias)
        conv3 = tf.nn.relu(pre_activation, name=scope.name)

    norm3 = tf.nn.lrn(conv3, 5, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
    pool3 = max_pool_2x2(conv3)

    with tf.variable_scope('local1') as scope:
        reshape = tf.reshape(pool3, [-1, 10 * 10 * 128])
        weights = weight_variable([10 * 10 * 128, 1024], "weights", wd=0.001)
        bias = bias_variable([1024], "biases")
        local1 = tf.nn.relu(tf.matmul(reshape, weights) + bias, name=scope.name)

    drop3 = tf.nn.dropout(local1, keep_prob)

    with tf.variable_scope('local2') as scope:
        weights = weight_variable([1024, 7], "weights", wd=0.001)
        bias = bias_variable([7], "biases")
        y_conv = tf.add(tf.matmul(drop3, weights), bias, name=scope.name)

    return y_conv, {}, {"h_conv1": conv1, "h_conv2": conv2, "keep_prob": keep_prob}


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
        mask = np.random.randint(0, labels.shape[0], (50,), np.int)

        X = images[mask]
        Y = labels[mask]
        feed_dict = {images_holder: X, labels_holder: Y, keys["keep_prob"]: values["keep_prob"]}

        _, loss_value = sess.run([train_step, l],
                                 feed_dict=feed_dict)

        if i % 10 == 0:
            saver.save(sess, "rect_model", global_step=i)

            feed_dict = {images_holder: test_images, labels_holder: test_labels, keys["keep_prob"]: 1.0}

            ac = sess.run(e, feed_dict=feed_dict)

            print('Step %d: loss = %.2f (%.3f )' % (i, loss_value, ac))


def evaluation(logits, labels_holder):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_holder, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, "float"))


def for_jaffe():
    with tf.Graph().as_default():
        sess = tf.InteractiveSession()

        # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True).train
        # images_tmp = mnist.images
        # labels = mnist.labels
        # images = np.reshape(images_tmp, [213, 24, 24, 3])
        images, labels = data_input.data_jaffe_load()

        images_holder = tf.placeholder(tf.float32, shape=(None, 24, 24, 3))
        labels_holder = tf.placeholder(tf.float32, shape=(None, 7))

        logits, params, mid_res = inference(images_holder)
        e = evaluation(logits, labels_holder)

        l = loss(logits, labels_holder)
        training(images, labels, l, e, sess=sess, reload=True)

        # 中间结果可视化
        mask = np.random.randint(0, 213, (50,), np.int)

        X = images[mask]
        Y = labels[mask]
        feed_dict = {images_holder: X, labels_holder: Y}

        #mprint mid_res["h_conv1"].eval(feed_dict=feed_dict).shape
        m = mid_res["h_conv2"].eval(feed_dict=feed_dict)[1,:,:,4]
        m /= np.max(m)
        m = cv2.resize(m, (0, 0), 0, 5.0, 5.0)
        cv2.imshow("1", m)
        # print m
        cv2.waitKey(0)


def for_me():
    """
    对我的表情数据进行表情识别
    :return:
    """
    with tf.Graph().as_default():
        sess = tf.InteractiveSession()

        images, labels = data_input.data_me_load()

        mask = np.arange(images.shape[0])
        np.random.shuffle(mask)
        train_mask = mask[0:-100]
        test_mask = mask[-100:]
        train_images = images[train_mask]
        train_labels = labels[train_mask]
        test_images = images[test_mask]
        test_labels = labels[test_mask]

        images_holder = tf.placeholder(tf.float32, shape=(None, 80, 80, 1))
        labels_holder = tf.placeholder(tf.float32, shape=(None, 7))

        logits, params, mid_res = inference(images_holder)
        e = evaluation(logits, labels_holder)

        l = loss(logits, labels_holder)
        keys = dict()
        values = dict()
        keys["keep_prob"] = mid_res["keep_prob"]
        values["keep_prob"] = 0.6
        training(train_images, train_labels, test_images, test_labels,keys,values, images_holder, labels_holder, l, e, sess=sess, reload=False, model_file="rect_model-90")

        i = 0

        while True:
            # 中间结果可视化
            i+=40
            i = i % labels.shape[0]
            mask = [i]

            X = images[mask]
            print X[0]
            feed_dict = {images_holder: X, keys["keep_prob"]: 1.0}
            print X
            print mid_res["y_conv"].eval(feed_dict=feed_dict).shape
            m = mid_res["y_conv"].eval(feed_dict=feed_dict)[0]
            print status[np.argmax(m)]
            print m
            print labels[mask]
            cv2.imshow("1", X[0])
            cv2.waitKey(0)

        cv2.waitKey(0)


def video_reg():
    """
    在视频帧当中读取数据进行表情识别
    """
    vc = cv2.VideoCapture(0)
    vc.open(0)
    if vc.isOpened():
        print "视频正常打开！"
    else:
        print "视频开启失败！"
        exit()

    with tf.Graph().as_default():
        sess = tf.InteractiveSession()

        images_holder = tf.placeholder(tf.float32, shape=(None, 80, 80, 1))
        labels_holder = tf.placeholder(tf.float32, shape=(None, 7))

        logits, params, mid_res = inference(images_holder)
        e = evaluation(logits, labels_holder)

        l = loss(logits, labels_holder)
        keys = dict()
        values = dict()
        keys["keep_prob"] = mid_res["keep_prob"]
        values["keep_prob"] = 1.0
        training([], [], [], [], keys, values, images_holder, labels_holder, l, e, sess=sess, reload=True, model_file="rect_model-1560")

        while True:
            ret, frame = vc.read()
            if ret == False:
                continue
            frame = cv2.flip(frame, 1)

            cv2.imwrite("tmp_file.jpg", frame)
            imfile = open("tmp_file.jpg", "rb")

            rect = fapi.get_face_rect(imfile)
            if not rect is None:
                ROI = frame[rect['top']: rect['top'] + rect['height'], rect['left']:rect['left'] + rect['width'], :]

                # 使用提取面部，统一大小为（80，80）
                ROI = cv2.resize(ROI, (80, 80))

                L_tmp = np.asarray(cv2.cvtColor(ROI, cv2.COLOR_RGB2LAB)[:, :, 0], dtype=np.uint8)
                L = np.zeros_like(L_tmp, dtype=np.uint8)
                cv2.equalizeHist(L_tmp, L)
                L_ = np.array(L, dtype=np.float32)
                cv2.normalize(L_, L_, 0, 1, cv2.NORM_MINMAX)

                file = "tmp_run.jpg"
                cv2.imwrite(file, L_ * 255)
                X = cv2.imread(file)
                L = np.asarray(cv2.cvtColor(X, cv2.COLOR_RGB2GRAY), dtype=np.float32)
                L_ = np.zeros_like(L, dtype=np.float32)
                cv2.normalize(L, L_, 0, 1.0, cv2.NORM_MINMAX)

                L_ = L_.reshape(( 80, 80, 1))

                feed_dict = {images_holder:[L_], keys["keep_prob"]:1.0}

                m = logits.eval(feed_dict=feed_dict)[0]

                cv2.imshow("i", L/255)
                print status[np.argmax(m)]
                print "自然状态:%s 恶心:%s 害怕:%s 快乐:%s 生气:%s 难过:%s 惊讶:%s " % (m[0], m[1], m[2], m[3], m[4], m[5],m[6])
            else:
                print ">>>"
            cv2.waitKey(30)


if "__main__" == __name__:
    # for_me()
    video_reg()
    pass


"""
TODO: 进行中间层数据的显示
"""