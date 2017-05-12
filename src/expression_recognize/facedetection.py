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
        points[k] = (landmarks[k]["x"]*1.0, landmarks[k]["y"]*1.0)
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


def draw_feature_point():
    p = "../../dataset/datashow/face/EP02_01f/"
    file_name_list = os.listdir(p)
    dir = p
    for file in file_name_list:
        if file[-3:] != "jpg":
            continue
        img = cv2.imread(dir+file)
        a = time.time()
        faces = get_faces(img)
        print time.time()-a
        landmarks = get_feature_points_fromimage(img)
        points = from_lanmark_to_points(landmarks)
        for i in points.keys():

            point = points[i]
            cv2.circle(img, (point[0], point[1]), 2, (0, 255.0, 0),)
        cv2.imshow("expression_recognize", img)
        cv2.waitKey(0)

def get_region_by_landmark():
    """
    landmark的划分
    :return:
    """
    regions=[]
    # 左侧内眉角
    left_int_brow = ["left_eyebrow_upper_middle", "left_eyebrow_lower_right_quarter", "left_eyebrow_right_corner", "brow_center", "brow_left_middle"]
    regions.append(left_int_brow)

    # 右侧内眉角
    right_int_brow = ["right_eyebrow_upper_middle", "right_eyebrow_lower_left_quarter", "right_eyebrow_left_corner", "brow_center", "brow_right_middle"]
    regions.append(right_int_brow)

    # 左侧外眉角
    left_out_brow = ["left_eyebrow_left_corner", "left_eyebrow_lower_left_quarter", "left_eyebrow_upper_middle", "brow_left_middle", "brow_left_left"]
    regions.append(left_out_brow)

    # 右侧外眉角
    right_out_brow = ["right_eyebrow_right_corner", "right_eyebrow_lower_right_quarter", "right_eyebrow_upper_middle", "brow_right_middle", "brow_right_right"]
    regions.append(right_out_brow)

    # 左侧脸颊
    left_chaw = ["contour_left2", "contour_left5", "mouth_left_corner", "nose_left", "nose_contour_left2"]
    regions.append(left_chaw)

    # 右侧脸颊
    right_chaw = ["contour_right2", "contour_right5", "mouth_right_corner", "nose_right", "nose_contour_right2"]
    regions.append(right_chaw)

    # 左侧嘴角上
    left_lip = ["nose_contour_left3", "lip_left_up",  "lip_left_left", "mouth_left_corner", "mouth_upper_lip_left_contour2"]
    regions.append(left_lip)

    # 右侧嘴角上
    right_lip = ["nose_contour_right3", "lip_right_up",  "lip_right_right", "mouth_right_corner", "mouth_upper_lip_right_contour2"]
    regions.append(right_lip)

    # 左侧嘴角下
    left_lip_down = ["mouth_left_corner", "lip_left_left", "lip_down_left", "mouth_lower_lip_left_contour2"]
    regions.append(left_lip_down)

    # 右侧嘴角下
    right_lip_down = ["mouth_right_corner", "lip_right_right", "lip_down_right", "mouth_lower_lip_right_contour2"]
    regions.append(right_lip_down)

    # 鼻子
    nose = ["nose_contour_left1", "nose_contour_left2", "nose_tip", "nose_contour_right2", "nose_contour_right1"]
    regions.append(nose)

    # 左侧外鼻翼

    # 右侧外鼻翼

    # 眉头

    # 上嘴唇
    up_lip = ["nose_contour_left3", "nose_contour_right3", "mouth_upper_lip_right_contour2", "mouth_upper_lip_left_contour2"]
    regions.append(up_lip)

    # 下嘴唇
    down_lip = ["lip_down_right", "mouth_lower_lip_right_contour2", "mouth_lower_lip_left_contour2", "lip_down_left"]
    regions.append(down_lip)

    # 左上眼帘
    left_eye = ["left_eyebrow_upper_left_quarter", "left_eye_upper_left_quarter", "left_eye_upper_right_quarter", "left_eyebrow_upper_right_quarter"]
    regions.append(left_eye)

    # 右上眼怜
    right_eye = ["right_eyebrow_upper_right_quarter", "right_eye_upper_right_quarter", "right_eye_upper_left_quarter", "right_eyebrow_upper_left_quarter"]
    regions.append(right_eye)

    return regions


def supply_landmark(points):
    """
    为了人脸划分需要补充一下特征点
    :param points:
    :return:
    """
    points["brow_center"] =((points["left_eyebrow_right_corner"][0]+points["right_eyebrow_left_corner"][0])/2,
                             (points["left_eyebrow_right_corner"][1] + points["right_eyebrow_left_corner"][1]) / 2 - 30)

    points["brow_left"] = (points["left_eyebrow_upper_right_quarter"][0],
                           points["left_eyebrow_upper_right_quarter"][1] - 30)

    points["brow_left_middle"] = (points["left_eyebrow_upper_middle"][0],
                                  points["left_eyebrow_upper_middle"][1] - 30)

    points["brow_left_left"] = (points["left_eyebrow_left_corner"][0],
                                points["left_eyebrow_left_corner"][1] - 30)

    points["brow_right"] = (points["right_eyebrow_upper_left_quarter"][0],
                            points["right_eyebrow_upper_left_quarter"][1] - 30)

    points["brow_right_right"] = (points["right_eyebrow_right_corner"][0],
                                  points["right_eyebrow_right_corner"][1] - 30)

    points["brow_right_middle"] = (points["right_eyebrow_upper_middle"][0],
                                   points["right_eyebrow_upper_middle"][1] - 30)

    points["lip_down_left"] = (points["contour_left9"][0],
                               points["contour_left9"][1] - 30)

    points["lip_down_right"] = (points["contour_right9"][0],
                                points["contour_right9"][1] - 30)

    points["lip_left_left"] = ((points["mouth_left_corner"][0] + points["contour_left5"][0]) / 2,
                                points["mouth_left_corner"][1])

    points["lip_right_right"] = ((points["mouth_right_corner"][0] + points["contour_right5"][0]) / 2,
                                points["mouth_right_corner"][1])

    points["lip_left_up"] = ((points["nose_contour_lower_middle"][0] + points["contour_left4"][0]) / 2,
                                points["nose_contour_lower_middle"][1])

    points["lip_right_up"] = ((points["nose_contour_lower_middle"][0] + points["contour_right4"][0]) / 2,
                                points["nose_contour_lower_middle"][1])


def draw_region(img, points):
    """
    绘制特征区域
    :return:
    """
    points = np.array([points], dtype=np.int32)
    cv2.polylines(img, points, True, (0,255,0))


def draw_region_mask(img, points):
    """
    绘制特征区域
    :return:
    """
    points = np.array([points], dtype=np.int32)
    cv2.fillConvexPoly(img, points, 1)


def face_region():
    """
    绘制特征区域
    :return:
    """
    p = "../../dataset/datashow/face/EP02_01f/"
    file_name_list = os.listdir(p)
    dir = p
    for file in file_name_list:
        if file[-3:] != "jpg":
            continue
        img = cv2.imread(dir+file)

        landmarks = get_feature_points_fromimage(img)
        points = from_lanmark_to_points(landmarks)
        supply_landmark(points)
        regions = get_region_by_landmark()
        points_list = []
        for i in regions:
            tmp_point = []
            for mark_name in i:
                tmp_point.append(points[mark_name])
            points_list.append(tmp_point)
            draw_region_mask(img, tmp_point)
        cv2.imshow("a", img)
        cv2.waitKey(0)
        print points_list


if "__main__" == __name__:
    face_region()

