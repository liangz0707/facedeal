# coding:utf-8
import os
import cv2
import numpy as np
import tensorflow as tf
import faceppapi as fapi


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
    file_name_list = os.listdir("../../dataset/expression_rect")
    dir = "../../dataset/expression_rect/"

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


def extract_face_rect():
    # 将人脸从图片中提取出来
    D = {"NE": 0, "DI": 1, "FE": 2, "HP": 3, "AN": 4, "SA": 5, "SU": 6}
    file_name_list = os.listdir("../../dataset/expression_image")
    dir = "../../dataset/expression_image/"

    rect_dir = "../../dataset/expression_rect/"

    for index, image_file in enumerate(file_name_list):
        if os.path.isdir(image_file):
            continue
        if not image_file[-3:] == "jpg":
            continue

        img_file = open(dir+image_file, "rb")
        image = cv2.imread(dir+image_file)

        # 使用face++找到面部矩形
        rect = fapi.get_face_rect(img_file, verbose=True)

        if rect == None:
            continue
        ROI = image[rect['top']: rect['top'] + rect['height'], rect['left']:rect['left'] + rect['width'], :]

        # 使用提取面部，统一大小为（80，80）
        ROI = cv2.resize(ROI, (80, 80))
        # 统一转换成Lab空间的L通道，数据范围为[0,1.0]的单通道浮点数图像
        L_tmp = np.asarray(cv2.cvtColor(ROI, cv2.COLOR_RGB2LAB)[:, :, 0], dtype=np.uint8)
        L = np.zeros_like(L_tmp, dtype=np.uint8)
        cv2.equalizeHist(L_tmp, L)
        L_ = np.array(L, dtype=np.float32)
        cv2.normalize(L_, L_, 0, 1, cv2.NORM_MINMAX)

        file = "%s%s%d.jpg" % (rect_dir, image_file[0:2], index)
        cv2.imwrite(file, L_ * 255)
        img_file.close()

if "__main__" == __name__:
    data_me_load()
    # extract_face_rect()

