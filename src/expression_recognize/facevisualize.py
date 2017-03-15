# coding:utf-8
import os

import cv2
import matplotlib.pyplot as plt
import numpy as NP
import numpy as np
from scipy import linalg as LA
import cPickle
import random
import sys
import math
sys.path.append("..")
from  motion_magnification.magnify import Magnify

D = {"NE": 0, "DI": 1, "FE": 2, "HP": 3, "AN": 4, "SA": 5, "SU": 6}

def makecolorwheel():
    colorwheel = []
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6
    for i in range(0, RY):
        colorwheel.append((255.0, 255.0 * i / RY, 0))
    for i in range(0, YG):
        colorwheel.append((255.0 - 255.0 * i / YG, 255.0, 0))
    for i in range(0, GC):
        colorwheel.append((0, 255.0, 255.0 * i / GC))
    for i in range(0, CB):
        colorwheel.append((0, 255.0 - 255.0 * i / CB, 255.0))
    for i in range(0, BM):
        colorwheel.append((255.0 * i / BM, 0, 255.0))
    for i in range(0, MR):
        colorwheel.append((255.0, 0, 255.0 - 255.0 * i / MR))
    return np.array(colorwheel, dtype=np.int)


def visual_flow(flow):
    """
    使用木塞尔颜色描述光流形态 将而为通道的光流，转换成三维通道的RGB值
    :param flow:
    :return:
    """
    shape = flow.shape
    color = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    colorwheel = makecolorwheel()
    maxrad = -1.0
    UNKNOWN_FLOW_THRESH = 400
    for i in range(shape[0]):
        for j in range(shape[1]):
            flow_at_point = flow[i, j]
            fx = flow_at_point[0]
            fy = flow_at_point[1]
            if np.fabs(fx) > UNKNOWN_FLOW_THRESH or np.fabs(fy) > UNKNOWN_FLOW_THRESH:
                continue
            rad = np.sqrt(fx * fx + fy * fy)
            if maxrad > rad:
                maxrad = maxrad
            else:
                maxrad = rad

    for i in range(shape[0]):
        for j in range(shape[1]):
            flow_at_point = flow[i, j]

            fx = flow_at_point[0] / maxrad
            fy = flow_at_point[1] / maxrad
            if np.fabs(fx) > UNKNOWN_FLOW_THRESH or np.fabs(fy) > UNKNOWN_FLOW_THRESH:
                color[i, j] = (0, 0, 0)
                continue
            rad = np.sqrt(fx * fx + fy * fy)
            angle = math.atan2(-fy, -fx) / cv2.cv.CV_PI
            fk = (angle + 1.0) / 2.0 * (len(colorwheel) - 1)
            k0 = int(fk)
            k1 = (k0 + 1) % len(colorwheel)
            f = fk - k0
            for b in range(3):
                col0 = colorwheel[k0][b] / 255.0
                col1 = colorwheel[k1][b] / 255.0
                col = (1 - f) * col0 + f * col1
                if rad <= 1:
                    col = 1 - rad * (1 - col)
                else:
                    col *= .75
                color[i,j][2 - b] = int(255.0 * col)
    return color


def draw_flow(im,flow,step=8):
    """
    在图片上绘制光流箭头
    :param im:
    :param flow:
    :param step:
    :return:
    """
    h,w = im.shape[:2]
    y,x = np.mgrid[step/2:h:step,step/2:w:step].reshape(2,-1)
    fx,fy = flow[y,x].T

    # create line endpoints
    lines = np.vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
    lines = np.int32(lines)

    # create image and draw
    vis = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    for (x1,y1),(x2,y2) in lines:
        cv2.line(vis,(x1,y1),(x2,y2),(0,255,0),1)
        cv2.circle(vis,(x1,y1),1,(0,255,0), -1)
    return vis


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
    evecs = evecs[:, idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return NP.dot(evecs.T, data.T).T, evals, evecs,idx


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


def landmark_feature():
    """
    读取ck+的特征点数据，计算稳定特征点，用于作为参考点，计算其他的特征，我们使用PCA来查看最稳定的特征
    :return:
    """
    file = open("../../dataset/ckplus_landmark_training_data.cpickle", "rb")
    train_landmark, train_label, test_landmark = cPickle.load(file)
    distense = np.zeros((train_landmark.shape[0], train_landmark.shape[1] * train_landmark.shape[1]))
    for i, plist in enumerate(train_landmark):
        a = 0
        for j, p in enumerate(plist):
            sub = p - plist
            sub[j, 1] = 0
            sub[j, 0] = 0
            mul = np.power(sub[:, 0] * sub[:, 0] + sub[:, 1] * sub[:, 1], 0.5)
            distense[i, a:a+68] = mul
            a = a + 68
    # Mat, evals, evecs, idx = PCA(distense, 4624)

    distense -= distense.mean(axis=0)
    # 计算协方差矩阵
    R = NP.cov(distense, rowvar=False)
    print R.shape
    # 计算协方差矩阵最大的值，按PCA的观点来看，就是方差最大的坐标轴就是第一个奇异向量，方差次大的坐标轴就是第二个奇异向量…。
    dd = R.diagonal()
    # 对方差的大小进行排列，查看最稳定的特征点对。首先自己和自己是最稳定的因为距离都是0
    l = np.argsort(dd)
    print l

    pairs = []
    for index, i in enumerate(l):
        # print "%d : %d -- %d" % (index, i % 68, i / 68)
        if index > 67 and i % 68 > i / 68:
            pairs.append((i % 68, i / 68))
    """
    这里计算相互的上面的点，刚好自己和自己是最稳定的
    TODO: 下一步可以尝试进行三角剖分，计算最稳定的三角形。
    """
    return pairs


def draw_landmarks(image, landmark, verbose=False):
    """
    ck 的方向和 casme2的点的顺序是反
    :param image:
    :param landmark:
    :param verbose:
    :return:
    """
    for index, p in enumerate(landmark):
        if verbose:
            cv2.putText(image, "%d" % index, (p[0], p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0))
        cv2.circle(image, (p[0], p[1]), 3, (255, 0, 0))


def ckplus_data_feature_vis(pairs, l=200, verbose=False):
    file = open("../../dataset/ckplus_all_training_data.cpickle", "rb")
    data = cPickle.load(file)

    # 选择一个图片和特征来进行显示
    landmark = data["train_landmark"][5]
    image = data["train_image"][5]
    draw_landmarks(image, landmark, True)
    xlen = len(pairs)
    l = np.min((l, xlen))
    l = np.max((1, l))

    #pairs.reverse()
    for index, i in enumerate(pairs):
        if index >= l:
            break
        cv2.line(image, (landmark[i[0]][0], landmark[i[0]][1]), (landmark[i[1]][0], landmark[i[1]][1]),
                 (0, 255 * (1 - index/1.0 / l), 255 * index/1.0 / l))
    cv2.imshow(" ", image)
    cv2.waitKey(0)


def degreefromvec(v1, v2):
    """
    通过cos 计算两个向量的夹角
    :param v1:
    :param v2:
    :return:
    """
    cos = 1.0 * np.dot(v1, v2) / np.sqrt(np.dot(v1, v1.T)) / np.sqrt(np.dot(v2, v2.T))
    return math.acos(cos) * 180


def squarefromvec(v1, v2):
    """
    通过cos 计算两个向量的面积
    :param v1:
    :param v2:
    :return:
    """
    squre = v1[0] * v2[0] - v1[1] * v2[1]
    return np.abs(squre)


def casme2_data_stastical_feature_vis():
    """
    进行统计特征的可视化
    :return:
    """
    D = {"happiness": 0, "others": 1, "disgust": 2, "repression": 3, "surprise": 4, "fear": 5, "sadness": 6}
    file = open("../../dataset/casme2_tiny_data.cpikle", "rb")
    data = cPickle.load(file)
    file.close()

    file = open("piar.cpickle", "rb")
    pairs = cPickle.load(file)
    file.close()

    # 显示人脸特征点
    landmark = data["landmarks"][0][0]
    image = data["images"][0][0]
    landmark = np.array(landmark)
    landmark = landmark[:, ::-1]
    draw_landmarks(image, landmark, True)
    cv2.imshow('img', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    这里选择第一个数据集，计算20-21， 22-23的眉间夹脚：
    0 happy 130 - 147.5    137.57  眉头舒展
    1 other 135 - 150.0    143.0  没有明显表情
    2 other 135 - 148      141   明显皱眉
    3 other 135 - 150      142   不明显
    4 other 136 - 148      141   皱眉
    5 other 125.0 - 145.   132   皱眉
    6 disgust  125 - 140.    134   皱眉
    7 disgust  125 - 140.    129   皱眉

    '''
    for F in range(9):
        # F = 1
        y = []
        x = range(len(data['images'][F]))
        print data["labels"][F]
        for i in x:
            l20 = data["landmarks"][F][i][20]
            l21 = data["landmarks"][F][i][21]
            l22 = data["landmarks"][F][i][22]
            l23 = data["landmarks"][F][i][23]
            v1 = l20 - l21
            v2 = l22 - l23
            #d = degreefromvec(v1, v2)
            d = squarefromvec(v1, v2)
            y.append(d)
            landmark = data["landmarks"][F][i]
            image = data["images"][F][i]
            landmark = np.array(landmark)
            landmark = landmark[:, ::-1]
            draw_landmarks(image, landmark)
            cv2.imshow("a:", image)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
        plt.plot(x, y)
        print np.mean(y)
        plt.show()


def casme2_data_feature_vis():
    '''
    从文件中取出数据
    '''
    D = {"happiness": 0, "others": 1, "disgust": 2, "repression": 3, "surprise": 4, "fear": 5, "sadness": 6}
    file = open("../../dataset/casme2_tiny_data.cpikle", "rb")
    data = cPickle.load(file)
    print "数据中的序列数量%d" % len(data["images"])
    F = 0
    print "当前表情：%s" % [data["labels"][F]]

    '''
    这部分内容，是对第一张图像和，峰值图像进行单帧迭代放大的结果，能够很容易的看出表情的状态
    '''
    for F in range(10):
        mg = Magnify()
        img1 = data["images"][F][0]
        img2 = data["images"][F][data["tops"][F]]
        print data["labels"][F]
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
        r, c = mg.magnify_frame(img1, img2)

        img1 = cv2.resize(img1, r.shape[0:2], None)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(r, cv2.COLOR_RGB2GRAY)

        # 这个光流方法是opencv自带的使用的是2003年的有点老
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, 0.5, 1, 3, 1, 3, 5, 1)
        cv2.imshow('Optical flow', draw_flow(gray1, flow * 5))
        cv2.imshow("b",c)
        cv2.imshow("flow", visual_flow(flow))
        cv2.waitKey(0)

    '''
    显示图像和特征点
    '''
    # image = data["images"][F][0]
    # for index in range(len(data["images"][F])):
    #     for i, p in enumerate(data["landmarks"][F][index]):
    #         cv2.circle(image, (p[1], p[0]), 1,  (255, 255, 0))
    #     cv2.imshow("a", image)
    #     cv2.waitKey(0)

if "__main__" == __name__:
    '''
    这一部分是对ckplus进行特征点关系分析
    '''
    # # 使用CK数据集计算稳定关联点对。
    # pairs = landmark_feature()
    # # 绘制出了特征点之间的稳定联系。
    # ckplus_data_feature_vis(pairs, l=100, verbose=True)
    # # 把特征点关系对保存起来
    # file = open("piar.cpickle","wb")
    # cPickle.dump(pairs, file)
    # file.close()

    file = open("piar.cpickle", "rb")
    pairs = cPickle.load(file)
    file.close()
    ckplus_data_feature_vis(pairs)
    """
    这一部分是对camse进行特征点关系分析
    """
    casme2_data_feature_vis()

    # casme2_data_stastical_feature_vis()
    pass