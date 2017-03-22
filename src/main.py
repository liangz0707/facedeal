# coding:utf-8

import matplotlib
matplotlib.use('TkAgg')

import data_collector.data_input as din
import expression_recognize.facevisualize as fv

import cv2
import matplotlib.pyplot as plt
import cPickle
import numpy as np
import img_toolkits.OpticalFlow_feature as of
import expression_recognize.facedetection as fd


def face_vis():
    train_image, train_label, test_image, test_image = din.load_kaggle_face_data(max_len=1000)

    Mat, evals, evecs = fv.feature_face(train_image, dim=5)

    print len(train_image)
    print Mat.shape
    print evals.shape
    print evecs.shape

    plt.figure(1)

    for i in range(len(train_image)):
        plt.plot(Mat[:, 0], Mat[:, 1], "b.")

        plt.plot(Mat[i, 0], Mat[i, 1], "r.")

        cv2.imshow("ex", train_image[i].reshape(48, 48))
        plt.show()


def ck_vis():
    '''
    人脸的样子进行PCA降维后，最开始 是全脸的样子，往后，就出现了对人的五官的权重的加深
    :return:
    '''
    ck_file = open("ck_faces_miniversion.cp", "rb")
    train_image, train_label, test_label = cPickle.load(ck_file)
    #train_image = cPickle.load(ck_file)
    Mat, evals, evecs = fv.feature_face(train_image, 10)
    print Mat.shape

    print evecs.shape
    plt.figure(1)

    color_map = ["r.", "g.", "b.", "c.", "m.", "y.", "k.", "r."]
    for i, m in enumerate(evecs.T):
        n = np.reshape(m, (48, 48))
        img = cv2.normalize(n, 0, 1, cv2.NORM_MINMAX)
        cv2.imshow("a", cv2.resize(img*255, (img.shape[0] * 5, img.shape[1]*5)))

        cv2.imshow("b", cv2.resize(train_image[0], (img.shape[0] * 5, img.shape[1]*5)))

        cv2.imshow("c", cv2.resize(train_image[0] * img, (img.shape[0] * 5, img.shape[1]*5)))
        cv2.waitKey(0)
    ''''''
    for i in range(len(Mat)):
        plt.plot(Mat[i, 2], Mat[i, 1], color_map[train_label[i]])

    plt.show()


def ck_training_data_trasform():

    faces, labels, test_faces = din.load_ckplus()
    print len(faces)
    print len(labels)
    print len(test_faces)

    print faces[0].shape

    gray_faces = []
    for face in faces:
        gray_faces.append(cv2.cvtColor(face, cv2.COLOR_RGB2GRAY))

    gray_tests = []
    for face in test_faces:
        gray_tests.append(cv2.cvtColor(face, cv2.COLOR_RGB2GRAY))

    ck_faces = open("ck_faces_full.cp", "wb")

    cPickle.dump((gray_faces, labels, gray_tests), ck_faces)


def flow_reader(file_path):
    """
    从FileSortage文件当中读取保存的数据
    :return:
        """
    U = np.asarray(cv2.cv.Load(file_path, cv2.cv.CreateMemStorage(), "U"))
    V = np.asarray(cv2.cv.Load(file_path, cv2.cv.CreateMemStorage(), "V"))
    return U, V


def test_affine_mat():
    feature_point_front = np.array([(1, 2), (3, 4), (3, 8)], dtype=np.float32)
    feature_point_back = feature_point_front * 1
    op = of.OpticalFlow()
    mat = op.calc_affine_mat(feature_point_front, feature_point_back)
    print mat
    pass


def aligen(image1,image2,U,V):
    pass


if __name__ == '__main__':
    """
    人脸对齐：
    1。读取光流，文件中的内容
    2。读取两帧图像
    3。读取第一帧图像的特征点
    4。计算由光流变到第二帧后的位置。
    5。计算仿射变换
    6。对第二张图像进行仿射变换来对齐
    """

    im1 = cv2.imread("../dataset/sub03/EP01_2/img51.jpg")
    im2 = cv2.imread("../dataset/sub03/EP01_2/img52.jpg")
    U, V = flow_reader("../dataset/sub03/EP01_2/reg_flow51.xml")
    vd = fd.get_feature_points_fromimage(im1)
    points = np.array(fd.from_lanmark_to_points(vd).values())

    # for i, point in enumerate(points):
    #     print point
    #     cv2.circle(im1, (point[0], point[1]) ,1, (0,255,0))
    #     cv2.putText(im1, str(i),(point[0]-10, point[1]) , cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0))
    # cv2.imshow("a",im1)
    # cv2.waitKey(0)

    """
    face++提取的特征点索引"""
    counter_index = [25,24,23,22,21,20,19,56,15,14,41,65,32,76,36,31,27,29,6]

    feature_pos = points[counter_index]
    # 通过光流计算feature_back
    feature_back = []
    for pos in feature_pos:
        new_pos = (pos[0] + U[pos[1]][pos[0]], pos[1] + V[pos[1]][pos[0]])
        print pos, new_pos
        feature_back.append(new_pos)
    fl = of.OpticalFlow()
    mat = fl.calc_affine_mat(np.array(feature_back, dtype=np.float32),np.array(feature_pos, dtype=np.float32))
    print mat
    A = cv2.warpPerspective(im1, mat.T, (im1.shape[1],im1.shape[0]))
    B = cv2.warpPerspective(im2, mat.T, (im1.shape[1],im1.shape[0]))
    cv2.imwrite("../dataset/sub03/EP01_2/img51_a.jpg",A)
    cv2.imwrite("../dataset/sub03/EP01_2/img52_a.jpg",B)



