# coding:utf-8
import os
import cv2
import numpy as np
import csv
import sys
sys.path.append("..")
import expression_recognize.facedetection as fd
import time


def from_num_to_vector(num, size):
    v = np.zeros((size))
    v[np.int(num)] = 1.0
    return v


def load_ckplus(face_shape=(48, 48)):
    """

    S001 -  S999_ 001 - 009_ 00000001 - 000000020

    :return:
    """
    dir = "../dataset/CK+/"
    image_count = 0
    label_count = 0
    land_count = 0

    train_image = []
    train_label = []
    test_image = []
    test_label = []

    for subject in xrange(999):
        for sqe in xrange(10):
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

                if os.path.isfile(image_name) != os.path.isfile(landmark_name):
                    print os.path.abspath(image_name)
                    print os.path.abspath(landmark_name)
                    print "landmark特征点和图像数据不对称"

                if os.path.isfile(image_name):
                    start = time.time()
                    tmp_img = cv2.imread(image_name)
                    faces = fd.get_faces(tmp_img)
                    end = time.time()
                    print "提取一个人脸图片的时间为：%sS" % (end - start)
                    if len(faces) != 0:
                        if label == None:
                            test_image.append(cv2.resize(faces[0], face_shape))
                        else:
                            train_image.append(cv2.resize(faces[0], face_shape))
                            train_label.append(label)


                if os.path.isfile(landmark_name):
                    land_count = land_count+1

                if os.path.isfile(label_name):
                    label_count = label_count+1

    return train_image, train_label, test_image


def load_kaggle_face_data(max_len = -1):
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


if "__main__" == __name__:
    train_image, train_label, test_image, test_label = load_kaggle_face_data()
    print len(train_image)
    print len(test_image)
    print train_image[1].shape

