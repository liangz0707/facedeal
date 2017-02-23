# coding:utf-8
import dlib
import cv2
import os
import time

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../dataset/shape_predictor_68_face_landmarks.dat")


def get_landmarks(image):
    """
    计算人脸的特征点
    :param image:
    :return:
    """
    dets = detector(image, 1)
    landmarks = []
    for i, d in enumerate(dets):
        landmarks.append(predictor(image, d))
    return landmarks


def get_faces(image):
    """
    把图像中的所有人脸都保存在一个faces的列表中，图像格式不变
    :param image:
    :return:
    """
    dets = detector(image, 1)
    faces = []
    for i, d in enumerate(dets):
        faces.append(image[d.top():d.bottom(), d.left():d.right()])
    return faces


if "__main__" == __name__:
    file_name_list = os.listdir("../../dataset/expression_image/DI/")
    dir = "../../dataset/expression_image/DI/"
    for file in file_name_list:
        if file[-3:] != "jpg":
            continue
        img = cv2.imread(dir+file)
        a = time.time()
        faces = get_faces(img)

        print time.time()-a
        landmarks = get_landmarks(img)
        for i in range(68):
            point = landmarks[0].part(i)
            cv2.circle(img, (point.x, point.y),2, (0, 1.0, 0))
        cv2.imshow("expression_recognize",img)
        cv2.waitKey(0)