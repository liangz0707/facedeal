# coding:utf-8

import cv2
from face import faceppapi
import time

if __name__ == '__main__':

    vc = cv2.VideoCapture(0)

    if vc.isOpened():
        print "video open success"
    else:
        print "video open false"

    imfile = None
    while True:
        a = time.time()
        ret, frame = vc.read()
        frame = cv2.flip(frame, 1)
        image = cv2.resize(frame, (frame.shape[1] / 5, frame.shape[0] / 5))

        cv2.waitKey(30)

        print "读取帧耗时：%s " % (time.time() - a)
        a = time.time()
        cv2.imwrite("tmp_file.jpg",  image)
        imfile = open("tmp_file.jpg", "rb")
        print "缓存图像耗时：%s " % (time.time() - a)
        a = time.time()
        obj = faceppapi.get_feature_point(imfile)
        if len(obj['faces']) != 0:
            if obj['faces'][0].has_key("landmark"):
                left = obj['faces'][0]['left_eye_pupil']
                right = obj['faces'][0]['landmark']['right_eye_pupil']
                cv2.circle(image, (right['x'], right['y']), 1, (0, 255, 0), -1)
                cv2.circle(image, (left['x'], left['y']), 1, (0, 255, 0), -1)

            rect = obj['faces'][0]['face_rectangle']

            cv2.rectangle(image, (rect['left'], rect['top']), (rect['left'] + rect['width'], rect['height'] + rect['top']), (0, 255, 0), 1)
        print "特征识别耗时：%s " % (time.time() - a)
        a = time.time()
        cv2.imshow("haha", image)
        cv2.waitKey(10)
