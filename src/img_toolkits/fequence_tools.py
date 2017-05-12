# -*- coding: utf-8 -*-
# Created by liangzh0707 on 2017/3/1
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import cv2


def flow_reader(file_path):
    """
    从FileSortage文件当中读取保存的数据
    :return:
        """
    U = np.asarray(cv2.cv.Load(file_path, cv2.cv.CreateMemStorage(), "U"))
    V = np.asarray(cv2.cv.Load(file_path, cv2.cv.CreateMemStorage(), "V"))
    return U, V


def read_image_list(path):
    """
    输入一个目录，返回所有的图像列表
    :return:
    """
    img_dist = os.listdir(path)
    img_list = []
    for image_file_name in img_dist:
        if image_file_name[-3:] != "jpg":
            continue
        img = cv2.imread(path + "/" + image_file_name)
        img_list.append(img)
    images_mat = np.array(img_list)
    return images_mat


def read_flow_list(path):
    """
    输入一个目录，返回所有的图像列表
    :return:
    """
    flow_dist = os.listdir(path)
    flow_list_U = []
    flow_list_V = []
    for flow_file_name in flow_dist:
        if flow_file_name[-3:] != "xml":
            continue
        U, V = flow_reader(path + "/" + flow_file_name)
        flow_list_U.append(U)
        flow_list_V.append(V)
    U_mat = np.array(flow_list_U)
    V_mat = np.array(flow_list_V)
    return U_mat, V_mat


def frequence_filter(feq):
    b_fft = np.fft.fft(feq)
    b_real = b_fft.real
    b_imag = b_fft.imag
    l = feq.shape[0]
    for i in range(len(b_real)):
        if ( (i >= 0.001 * l and i <0.11 * l) or (i > 0.94 * l and i < 0.97 * l)):
            b_real[i] = b_real[i] * 50
            b_imag[i] = b_imag[i] * 50
        else:
            b_real[i] = 0 * b_real[i] * 0
            b_imag[i] = b_imag[i] * 0

    b_fft.real = b_real
    b_fft.imag = b_imag
    return np.fft.ifft(b_fft).real


def frequence_test():
    point = []
    a_1 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    a_2 = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
    b_fft = np.fft.fft(a_1 + a_2)

    b_real = b_fft.real
    b_img = b_fft.imag
    b_real[6] += 30

    b_fft.real = b_real

    a_r = np.fft.ifft(b_fft).real

    plt.plot(range(10), a_r)
    plt.plot(range(10), a_1 + a_2)

    plt.show()

    print a_r


if __name__ == '__main__':
    pdir_img = "../../dataset/datashow/face_aligen/EP02_01f/"
    pdir_flow = "../../dataset/datashow/flow_aligen/EP02_01f/"
    tmp_dir = "../../dataset/tmp/"
    tmp_dir = "../../dataset/tmp/"

    np.zeros((1000, 1000))
    # img_mat = read_image_list(pdir_img)
    U_mat, V_mat = read_flow_list(pdir_flow)
    feq = np.sqrt(np.power(V_mat, 2) + np.power(U_mat, 2))[:, 30: -30, 30:-30]
    size = feq.shape[1:]
    l = feq.shape[0]

    for i in range(size[0]):
        for j in range(size[1]):
            feq[:, i, j] = frequence_filter(feq[:, i, j])
            # pass

    for i in range(l):
        img = feq[i, :, :]
        img = np.abs(img)
        cv2.imwrite("../../dataset/tmp/%d.jpg" % i, img*5)
    # print time_line
    # plt.plot(range(len(time_line)), time_line)
    # plt.show()




