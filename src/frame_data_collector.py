# coding:utf-8
import cv2
import numpy as np
import os

"""
    表情采集，采集一个人的7中表情状态。
"""
def collect_expression_image():
    vc = cv2.VideoCapture(0)
    if vc.isOpened():
        print "video open success"
    else:
        print "video open false"
        return
    e = "FE"
    i = 0
    while True:
        ret, frame = vc.read()
        frame = cv2.flip(frame, 1)
        image = cv2.resize(frame, (frame.shape[1] / 5, frame.shape[0] / 5))

        cv2.imwrite("../dataset/expression_image/FE/%s%d.jpg" % (e, i), frame)
        cv2.imshow("face", image)
        if cv2.waitKey(30) > 30:
            break

        i += 1

if "__main__" == __name__:
    collect_expression_image()


