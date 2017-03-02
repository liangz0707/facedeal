# coding:utf-8
import data_collector.data_input as din
import expression_recognize.facevisualize as fv
import matplotlib
matplotlib.use('TkAgg')
import cv2
import matplotlib.pyplot as plt
import cPickle
import numpy as np

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


def gabor_fn(sigma, theta, Lambda, psi, gamma):
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 3
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    return gb


if __name__ == '__main__':
    # ck_training_data_trasform()
    # ck_vis()
    cu = cv2.getGaussianKernel(10, 2)
    print cu
    filter = np.multiply(cu,cu.T)
    gus = cv2.normalize(filter,None, 0 ,1,cv2.NORM_MINMAX)
    kern = cv2.getGaborKernel((100, 100), 15.0, 3.14/2, 1.5, 10, 0, ktype=cv2.CV_32F)

    image = cv2.imread("../dataset/S067_002_00000006.png")
    cv2.imshow("aa", image)
    image_bluter = cv2.filter2D(image, cv2.CV_8UC3, kern)
    cv2.imshow("gus",kern)
    cv2.waitKey(0)



