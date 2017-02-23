# coding:utf-8
#
# 主要用来采集自己的人脸数据，需要有摄像头
# D = {"NE": 0, "DI": 1, "FE": 2, "HP": 3, "AN": 4, "SA": 5, "SU": 6}
#
import cv2
import numpy as np
import os
import src.expression_recognize.facedetection as fd
import src.expression_recognize.faceppapi as fapi


def extract_face_rect(face_image_dir, tag, target_dir, face_shape, use_faceplusplus=True):
    """
    将一个目录下的图片读取出来，并且制定这个目录的表情标签
    :param face_image_dir: 目录
    :param tag: 标签
    :param target_dir: 提取出的人脸图片保存的位置
    :param use_faceplusplus: 是否使用face++的api 或使用dlib
    :param face_shape: 人脸大小
    :return:
    """
    dir = face_image_dir
    file_name_list = os.listdir(dir)

    rect_dir = target_dir

    for index, image_file in enumerate(file_name_list):
        if os.path.isdir(image_file):
            continue
        if not image_file[-3:] == "jpg":
            continue
        img_file = open(dir+image_file, "rb")
        image = cv2.imread(dir+image_file)

        if not use_faceplusplus:
            faces = fd.get_faces(image)
            if len(faces) == 0:
                continue
            ROI = cv2.resize(faces[0], (80, 80))
        else:
            rect = fapi.get_face_rect(img_file, verbose=True)  # 使用face++找到面部矩形

            if rect is None:
                continue

            ROI = image[rect['top']: rect['top'] + rect['height'], rect['left']:rect['left'] + rect['width'], :]
            ROI = cv2.resize(ROI, face_shape)  # 使用提取面部，统一大小为（face_shape

        # 统一转换成Lab空间的L通道，数据范围为[0,1.0]的单通道浮点数图像
        L_tmp = np.asarray(cv2.cvtColor(ROI, cv2.COLOR_RGB2LAB)[:, :, 0], dtype=np.uint8)
        L = np.zeros_like(L_tmp, dtype=np.uint8)
        cv2.equalizeHist(L_tmp, L)
        L_ = np.array(L, dtype=np.float32)
        cv2.normalize(L_, L_, 0, 255, cv2.NORM_MINMAX)

        file = "%s%s%d.jpg" % (rect_dir, tag, index)
        cv2.imwrite(file, L_)
        img_file.close()


def collect_expression_image(expression, file_dir="../dataset/expression_image/HP/", zoom=0.2):
    """
        表情采集，采集一个人的7中表情状态。
        :param expression: 表情标签 D = {"NE": 0, "DI": 1, "FE": 2, "HP": 3, "AN": 4, "SA": 5, "SU": 6}
        :param file_dir: 文件目录
        :param zoom: 图片缩放因子
    """
    vc = cv2.VideoCapture(0)
    if vc.isOpened():
        print "摄像头成功打开"
    else:
        print "摄像头开启失败"
        return

    i = 0
    while True:
        ret, frame = vc.read()
        frame = cv2.flip(frame, 1)
        image = cv2.resize(frame, (frame.shape[1] * zoom, frame.shape[0] * zoom))

        cv2.imwrite("%s%s%d.jpg" % (file_dir, expression, i), frame)
        cv2.imshow("expression_recognize", image)
        if cv2.waitKey(30) > 30:
            break
        i += 1


if "__main__" == __name__:
    #  collect_expression_image()
    #  extract_face_rect()
    pass
    pass


