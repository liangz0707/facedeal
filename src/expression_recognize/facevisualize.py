# coding:utf-8
import tensorflow as tf
import data_input
import numpy as np
import time
import cv2
import faceppapi as fapi
import os
import scipy as sci
import numpy as NP
from scipy import linalg as LA
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

D = {"NE": 0, "DI": 1, "FE": 2, "HP": 3, "AN": 4, "SA": 5, "SU": 6}


def PCA(data, dims_rescaled_data=2):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """
    m, n = data.shape
    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = NP.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric,
    # the performance gain is substantial
    evals, evecs = LA.eigh(R)
    # sort eigenvalue in decreasing order
    idx = NP.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return NP.dot(evecs.T, data.T).T, evals, evecs


def feature_face():
    """
    显示特征脸的PCA结果，我们发现主要的区分还是建立在人脸上
    :return:
    """
    file_name_list = os.listdir("../../dataset/expression_rect")
    rect_dir = "../../dataset/expression_rect/"
    image_mat = np.zeros((len(file_name_list), 6400))
    for i, file in enumerate(file_name_list):
        if file[-3:] != 'jpg':
            continue
        image = cv2.imread(rect_dir + file)

        gray = np.asarray(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), dtype=np.uint8)
        image_mat[i, :] = gray.reshape((80 * 80))

    Mat, evals, evecs = PCA(np.array(image_mat,dtype=np.float32).T, 4)

    plt.figure(1)
    points = image_mat.dot(Mat)

    for i in range(1000):
        plt.plot(points[:, 2], points[:, 3], "b.")

        plt.plot(points[i, 2], points[i, 3], "r.")

        cv2.imshow("ex", image_mat[i].reshape(80, 80)/255)
        plt.show()


if "__main__" == __name__:
    feature_face()