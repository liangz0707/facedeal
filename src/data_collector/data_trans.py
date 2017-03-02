# -*- coding: utf-8 -*-
# Created by liangzh0707 on 2017/3/1

import cv2
import os


def from_image2video(dir, video_name):
    file_names = os.listdir(dir)
    video_path = os.path.join(dir, video_name)
    if os.path.exists(video_path):
        os.remove(video_path)
    images = []
    for file_name in file_names:
        file_path = os.path.join(dir, file_name)
        if not os.path.isfile(file_path):
            continue
        tmp_img = cv2.imread(file_path)
        images.append(tmp_img)

    assert len(images) > 0
    w = images[1].shape[1]
    h = images[0].shape[0]
    vw = cv2.VideoWriter(video_path, -1, 30, (w, h), True)

    for img in images:
        vw.write(img)
    vw.release()

if __name__ == '__main__':
    from_image2video("../../dataset/001","face.avi")
    pass
