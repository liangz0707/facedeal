# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def basic_1d_magnification():
    """
    最基本的1维的欧拉变形放大展示
    :return:
    """
    x = np.linspace(1, 10, 100)
    y = np.cos(x)
    t = 0.3
    dx = np.array(y)

    for i in range(1,len(x)):
        dx[i] = (y[i] - y[i - 1]) / (x[i] - x[i-1])
    y_t = np.cos(x + t)  # t = 0.2
    y_t_a = y + 1.9 * t * dx
    plt.figure(figsize=(8, 4))
    plt.plot(x, y_t, "b--", label="y_after")
    plt.plot(x, y, "r--", label="y_before")
    # plt.plot(x, dx, "y-", label="grad")
    plt.plot(x, y_t_a, "b-", label="y_magnified")
    plt.legend()
    plt.show()
    plt.show()


if "__main__" == __name__:
    basic_1d_magnification()





