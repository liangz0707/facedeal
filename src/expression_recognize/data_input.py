# coding:utf-8
import os
import cv2
import numpy as np
import tensorflow as tf
import faceppapi as fapi
import csv
import cPickle
import sys

def load_kaggle_face_data():
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
        if c == 20:
            break
        if line[2] == "Training":
            train_label.append(line[0])
            img = np.array([np.int(i) for i in line[1].split(' ')], dtype=np.uint8).reshape((48, 48))
            train_image.append(img)

        else:
            test_label.append(line[0])
            img = np.array([np.int(i) for i in line[1].split(' ')], dtype=np.uint8).reshape((48, 48))
            test_image.append(img)
    return train_image, train_label, test_image, test_label


def data_jaffe_load():
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


def data_me_load():
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
    #train_image, train_label, test_image, test_label = load_kaggle_face_data()
    pass
    # extract_face_rect()

