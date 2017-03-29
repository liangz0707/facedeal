# coding:utf-8
import dlib
import cv2
import os
import time
import numpy as np

import requests
import json
import cv2


def get_feature_point(image_file, verbose=False):
    """
    通过人脸图像获取举行位置和全部83个特征点
    :param image:  通过'rb'模式打开的二进制图片文件
    :return:
    """

    url = "https://api-cn.faceplusplus.com/facepp/v3/detect"
    param = {
        "api_key": "H8qZAoXMbP4X96zmoP35WHF4ZX_d4iLN",
        "api_secret": "VC2ztQ4ME5SnVDwOWO4s-uBQdH1r83Cp",
        "return_landmark": 1
    }
    files = {'image_file': ('image_file', image_file, 'multipart/form-data')}
    response = requests.post(url, data=param, files=files)

    if verbose is True:
        print response.content

    jd = json.JSONDecoder()
    if len(jd.decode(response.content)['faces']) == 0:
        return None, None
    else:
        return jd.decode(response.content)['faces'][0]['landmark'], jd.decode(response.content)['faces'][0]['face_rectangle']


def from_lanmark_to_points(landmarks):
    """
    将图片的标准点格式转换成points格式
    :return:
    """
    points = dict()
    for k in landmarks:
        points[k] = (landmarks[k]["x"], landmarks[k]["y"])
    return points


def get_feature_points_fromimage(image, verbose=False):
    """
    通过图片获取特征点，通过人脸图像获取举行位置和全部83个特征点，先保存成文件，然后再上传这个文件
    :param image:  numpy格式的图片
    :return:
    """
    cv2.imwrite("file_tmp.jpg", image)
    img_file = open("file_tmp.jpg", "rb")

    url = "https://api-cn.faceplusplus.com/facepp/v3/detect"
    param = {
        "api_key": "H8qZAoXMbP4X96zmoP35WHF4ZX_d4iLN",
        "api_secret": "VC2ztQ4ME5SnVDwOWO4s-uBQdH1r83Cp",
        "return_landmark": 1
    }
    files = {'image_file': ('image_file', img_file, 'multipart/form-data')}
    response = requests.post(url, data=param, files=files)

    if verbose is True:
        print response.content

    jd = json.JSONDecoder()
    if len(jd.decode(response.content)['faces']) == 0:
        return None
    else:
        return jd.decode(response.content)['faces'][0]['landmark']


def get_face_rect(image, verbose=False):
    """
    检测人脸所在的矩形框
    :param image:
    :param verbose:
    :return:
    """
    url = "https://api-cn.faceplusplus.com/facepp/v3/detect"
    param = {
        "api_key": "H8qZAoXMbP4X96zmoP35WHF4ZX_d4iLN",
        "api_secret": "VC2ztQ4ME5SnVDwOWO4s-uBQdH1r83Cp",
        "return_landmark": 0
    }
    files = {'image_file': ('image_file', image, 'multipart/form-data')}
    response = requests.post(url, data=param, files=files)

    if verbose is True:
        print response.content

    jd = json.JSONDecoder()
    if len(jd.decode(response.content)['faces']) == 0:
        return None
    else:
        return jd.decode(response.content)['faces'][0]['face_rectangle']


def from_filename2face(file_name, face_size, verbose=False):

    file_src = open(file_name, "rb")
    image = cv2.imread(file_name)
    rect = get_face_rect(file_src, verbose=verbose)
    if rect is None:
        return None
    else:
        face = image[rect['top']: rect['top'] + rect['height'], rect['left']:rect['left'] + rect['width'], :]
        face = cv2.resize(face, face_size)
        return face

#detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor("../../dataset/model/shape_predictor_68_face_landmarks.dat")


def get_landmarks(image):
    """
    计算人脸的特征点
    :param image:
    :return:
    """
    dets = detector(image, 1)
    landmarks = []
    for i, d in enumerate(dets):
        shape = predictor(image, d)
        landmarks.append([np.array([ shape.part(index).y - dets[0].top(), shape.part(index).x - dets[0].left()]) for index in range(68)])
    return landmarks[0], (dets[0].left(), dets[0].top(), dets[0].right(), dets[0].bottom())


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