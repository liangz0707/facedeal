# coding:utf-8

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
    return jd.decode(response.content)


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
