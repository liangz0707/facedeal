# coding:utf-8
import cv2
import numpy as np


def points_compare(point1, point2):
    """
    point1<point2
    为真
    :param point1:
    :param point2:
    :return:
    """
    if point1[0] < point2[0]:
        return True
    elif point1[0] == point2[0]:
        return point1[1] < point2[1]
    else:
        return False


def sort_point(index_list, points):
    """
    根据实际点的位置对索引进行排序，方便使用扫描线算法计算点的位置。
    :return:
    """
    for ind in index_list:
        if not points_compare(points[ind[0]], points[ind[1]]):
            ind[0], ind[1] = ind[1], ind[0]
        if not points_compare(points[ind[0]], points[ind[2]]):
            ind[0], ind[2] = ind[2], ind[0]
        if not points_compare(points[ind[1]], points[ind[2]]):
            ind[1], ind[2] = ind[2], ind[1]


def compute_affine_mat(triangles, points_src, points_dst):
    """
    计算变换矩阵
    :param
    triangles 当中保存了所有的三角形索引
    points_src 和 points_dst 分别是两套数据点
    :return:
    """
    affines = []
    for T in triangles:
        src = np.array([np.array(points_src[T[0]]),
                        np.array(points_src[T[1]]),
                        np.array(points_src[T[2]])], dtype=np.float32)

        dst = np.array([np.array(points_dst[T[0]]),
                        np.array(points_dst[T[1]]),
                        np.array(points_dst[T[2]])], dtype=np.float32)

        affines.append(cv2.getAffineTransform(src, dst))
    return affines


def draw_mask(triangles, points, mask_shape):
    """
    绘制涂层遮罩，没有必要算每个三角形的变形通过遮罩赋值
    points是目标
    :return:
    """
    masks = []
    for T in triangles:
        mat = np.zeros(mask_shape)
        pos = np.array([np.array(points[T[0]]),
                        np.array(points[T[1]]),
                        np.array(points[T[2]])], dtype=np.int32)
        cv2.fillPoly(mat, [pos], (1, 1, 1))
        masks.append(mat)
    return masks


def combine_image(images, mask, shape):
    """
    将变形后的内容通过mask组合
    :return:
    """
    result = np.zeros(shape)
    M = np.zeros(shape)  # 添加这个M是为了防止mask产生重叠
    for i, m in zip(images, mask):
        result = result + i * m
        M = m + M
    return result / M

if "__main__" == __name__:
    """
    对已知剖分结果的三角面进行变形
    """

    mat = cv2.imread("1.jpg")
    shape = mat.shape
    height = shape[1]
    width = shape[0]
    points_src = np.array([(0, 0), (width, 0), (0, height), (width / 2, height / 2), (width, height)])
    points_dst = np.array([(0, 0), (width, 0), (0, height), (width / 3, height / 3), (width, height)])
    triangles = np.array([[0, 1, 3], [0, 2, 3], [2, 3, 4], [1, 3, 4]])
    affine_mats = compute_affine_mat(triangles, points_src, points_dst)

    warp = []
    for M in affine_mats:
        I = cv2.warpAffine(mat, M, shape[0:2])
        warp.append(I)

    masks = draw_mask(triangles, points_dst, shape)

    result = combine_image(warp, masks, shape)

    cv2.imshow("dst", result/255)

    cv2.imshow("src", mat)
    cv2.waitKey(0)
