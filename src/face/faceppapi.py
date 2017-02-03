# coding:utf-8

import requests
import json


def get_feature_point(image, verbose=False):
    """
    通过人脸图像获取举行位置和全部83个特征点
    :param image:  通过'rb'模式打开的二进制图片文件
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
    return jd.decode(response.content)
