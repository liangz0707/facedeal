# coding:utf-8
import os

import cv2
import matplotlib
import numpy as NP
import numpy as np
from scipy import linalg as LA


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


def feature_face(images, dim = 10):
    """
    显示特征脸的PCA结果，我们发现主要的区分还是建立在人脸上
    :return:
    """
    shape = images[0].shape
    cols = 0
    rows = len(images)
    if len(shape) == 1:
        cols = 1
    elif len(shape) == 2:
        cols = shape[0] * shape[1]
    elif len(shape) == 3:
        cols = shape[0] * shape[1] * shape[2]
    else:
        print "数据纬度异常"
        return

    image_mat = np.zeros((rows, cols))
    for i, image in enumerate(images):
        vector = np.reshape(image, cols)
        image_mat[i, :] = vector

    Mat, evals, evecs = PCA(np.array(image_mat, dtype=np.float32), dim)
    return Mat, evals, evecs


if "__main__" == __name__:
    pass