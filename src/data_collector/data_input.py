# coding:utf-8
import os
import cv2
import numpy as np
import csv
import sys
sys.path.append("..")
import expression_recognize.facedetection as fd
import time

import cPickle


def from_num_to_vector(num, size):
    v = np.zeros((size))
    v[np.int(num)] = 1.0
    return v


def load_ckplus(n_sub=999):
    """

    S001 -  S999_ 001 - 009_ 00000001 - 000000020

    :return:
    """
    dir = "../../dataset/CK+/"
    image_count = 0
    label_count = 0
    land_count = 0

    train_image = []
    train_label = []
    test_image = []
    train_landmark = []
    test_landmark = []

    for subject in xrange(n_sub):
        for sqe in xrange(10):
            # 读取序列的标签
            label_dir = "%sEmotion/S%03d/%03d" % (dir, subject, sqe)
            if not os.path.isdir(label_dir):
                continue
            file_list = os.listdir(label_dir)
            if len(file_list) == 0:
                label = None
            else:
                f = open("%s/%s" % (label_dir, file_list[0]))
                label = int(float(f.readline()[:5]))

            for frame in xrange(20):
                content = "cohn-kanade-images"
                image_name = "%s%s/S%03d/%03d/S%03d_%03d_%08d.png" % (dir, content,  subject, sqe,subject, sqe, frame)
                content = "Landmarks"
                landmark_name = "%s%s/S%03d/%03d/S%03d_%03d_%08d_landmarks.txt" % (dir, content,  subject, sqe, subject, sqe, frame)
                content = "Emotion"
                label_name = "%s%s/S%03d/%03d/S%03d_%03d_%08d_emotion.txt" % (dir, content, subject, sqe,subject, sqe, frame)

                # 查看特征点和图像是不是一一对应
                if os.path.isfile(image_name) != os.path.isfile(landmark_name):
                    print os.path.abspath(image_name)
                    print os.path.abspath(landmark_name)
                    print "landmark特征点和图像数据不对称"

                if os.path.isfile(image_name):
                    # 读取图片
                    tmp_img = cv2.imread(image_name)
                    if label == None:
                        test_image.append(tmp_img)
                    else:
                        train_image.append(tmp_img)
                        train_label.append(label)

                    # 读取图片对应的特征点位置
                    file = open(landmark_name, "rb")
                    lines = file.readlines()
                    points = []
                    for line in lines:
                        line = line.replace("\n", "").strip()
                        p = line.split(" ")
                        point = np.array([np.float32(p[0]), np.float32(p[3])])
                        points.append(point)
                    file.close()
                    if len(points) == 0:
                        print image_name
                        print label
                        continue
                    if label == None:
                        test_landmark.append(np.array(points))
                    else:
                        train_landmark.append(np.array(points))

                if os.path.isfile(landmark_name):
                    land_count = land_count + 1

                if os.path.isfile(label_name):
                    label_count = label_count + 1

    data = {"train_image":train_image, "train_label":train_label, "train_landmark": train_landmark,
            "test_image": test_image, "test_landmark":test_landmark}

    return data


def load_kaggle_face_data(max_len=-1):
    """emotion, pixels, Usage
    """
    file = open('../../dataset/fer2013/fer2013.csv', 'rb')
    file.readline()
    reader = csv.reader(file)
    train_image = []
    train_label = []
    test_image = []
    test_label = []
    c = 1
    for line in reader:
        c = c + 1
        if c == max_len:
            break
        if line[2] == "Training":
            train_label.append(from_num_to_vector(line[0], 7))
            img = np.array([np.int(i) for i in line[1].split(' ')], dtype=np.uint8).reshape((48, 48, 1))
            train_image.append(img)

        elif line[2] == "PrivateTest" or line[2] == "PublicTest":
            test_label.append(from_num_to_vector(line[0], 7))
            img = np.array([np.int(i) for i in line[1].split(' ')], dtype=np.uint8).reshape((48, 48, 1))
            test_image.append(img)
        else:
            print line[2]
    return train_image, train_label, test_image, test_label


def load_jaffe_data():
    """
    XX(两个字母名字).YY(两个字母，情感)Z(一个数字，情感标号).NN(一个数字，图片编号).tiff
    :return:
    """
    D = {"NE": 0, "DI": 1, "FE": 2, "HA": 3, "AN": 4, "SA": 5, "SU": 6}
    file_name_list = os.listdir("../../dataset/jaffe")
    dir = "../../dataset/jaffe/"

    y = []
    x = []

    for file_name in file_name_list:
        tag = file_name.split('.')
        if len(tag) < 3:
            continue
        c = D[tag[1][0:2]]
        Y = np.zeros([7], np.float32)
        Y[c] = 1.0
        X = cv2.resize(cv2.imread(dir+file_name), (24, 24))
        y.append(Y)
        x.append(X)
    y = np.asarray(y)
    x = np.asarray(x)
    print y.shape
    print x.shape
    return x, y


def load_my_face_data():
    """
    XX(两个字母名字).YY(两个字母，情感)Z(一个数字，情感标号).NN(一个数字，图片编号).tiff
    :return:
    """
    D = {"PE": 0, "DI": 1, "FE": 2, "HP": 3, "AN": 4, "SA": 5, "SU": 6}
    file_name_list = os.listdir("../../dataset/expression_rect2")
    dir = "../../dataset/expression_rect2/"

    print "文件夹中图片数量:%d" % len(file_name_list)
    y = []
    x = []
    num = 0
    for file_name in file_name_list:
        tag = file_name[0:2]
        if not tag in D.keys():
            continue
        num += 1
        Y = np.zeros([7], np.float32)
        Y[D[tag]] = 1.0
        X = cv2.resize(cv2.imread(dir+file_name), (80, 80))

        # 统一转换成Lab空间的L通道，数据范围为[0,1.0]的单通道浮点数图像
        L = np.asarray(cv2.cvtColor(X, cv2.COLOR_RGB2GRAY), dtype=np.float32)
        L_ = np.zeros_like(L, dtype=np.float32)
        cv2.normalize(L, L_, 0, 1.0, cv2.NORM_MINMAX)

        y.append(Y)
        x.append(L_)

    y = np.asarray(y)
    x = np.asarray(x)
    x = x.reshape((num, 80, 80, 1))
    print x.shape
    print "训练数据数量:%d" % len(x)
    return x, y


def load_casme2_all():
    """
    目录结构 sub01 - sub26/ EP 。。。 ／ *.jpg
    标记等其他信息在CASME2-coding-20140508.csv当中
    label = happiness, others, disgust, repression, surprise
    读取出全部的图像
    这部分可以用来查看固定不变的特征在什么位置
    :return:
    返回的数据变成一个list data  1-26,
    """
    root = '../../dataset/CASME2/CASME2_RAW_selected'
    D = {"happiness": 0, "others": 1, "disgust": 2, "repression" : 3, "surprise": 4, "fear":5,"sadness":6}
    labels = []
    files = []
    subs = []
    images = []
    landmarks = []
    tops = []

    file = open("../../dataset/CASME2/CASME2-coding-20140508.csv", "rb")

    lines = file.readlines()[1:]
    a = 0
    for line in lines:
        a = a + 1
        if a > 10:
            break
        line = line.strip()
        meta = line.split(",")
        print meta
        sub = int(meta[0])
        file_name = meta[1]
        begin = int(meta[3])
        top = int(meta[4]) - begin  # 相对位置
        end = int(meta[5])
        label = D[meta[8]]
        sequence = []
        landmark = []
        print "已完成%f%%" % (1.0 * a / len(lines) * 100)
        for i in range(begin,end):
            path = "%s/sub%02d/%s/img%d.jpg" % (root, sub, file_name, i)
            assert os.path.exists(path)
            image = cv2.imread(path)
            gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
            L, R = fd.get_landmarks(image)
            landmark.append(np.array(L))
            sequence.append(gray[R[1]:R[3], R[0]:R[2]])
        subs.append(sub)
        tops.append(top)
        labels.append(label)
        files.append(file_name)
        images.append(sequence)
        landmarks.append(landmark)
    print len(images), len(landmarks), len(labels), len(subs)
    return {"images": images, "landmarks": landmarks, "labels": labels, "sub": subs, "files": files, "tops": tops}


if "__main__" == __name__:
    '''
    # 用来读取CK+数据集的landmark数据，得到的是ndarray数据(4849, 68, 2)
    train_landmark, train_label, test_landmark = load_ckplus_landmark()
    file = open("ckplus_landmark_training_data.cpickle", "wb")
    cPickle.dump((np.array(train_landmark), train_label, np.array(test_landmark)), file)
    file.close()
    print len(train_landmark), len(train_label), len(test_landmark)

    print np.array(train_landmark).shape
    print np.array(test_landmark).shape
    '''

    '''
    load_ckplus:读取全套的数据
    data = load_ckplus(n_sub=10)
    file = open("../../data/ckplus_all_training_data.cpickle", "wb")
    cPickle.dump(data, file)
    file.close()
    for i in data.keys():
        print "%s : %d" % (i, len(data[i]))
    '''

    '''
    读取casme2中的数据
    转化成了灰度值，计算了特征点等内容
    data = load_casme2_all()
    file = open("../../dataset/casme2_tiny_data.cpikle", "wb")
    cPickle.dump(data, file)
    file.close()
    '''

    pass
