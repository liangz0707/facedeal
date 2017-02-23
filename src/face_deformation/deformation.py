# coding:utf-8
import cv2
import numpy as np
import sys
sys.path.append("..")
import expression_recognize.faceppapi as fapi
from scipy.spatial import Delaunay
import expression_recognize.data_input as fdata


def compute_affine_mat(triangles, points_src, points_dst):
    """
    计算仿射变换矩阵
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


def draw_masks(triangles, points, mask_shape):
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


def draw_mask(triangles, points, mask_shape):
    """
    绘制涂层遮罩，没有必要算每个三角形的变形通过遮罩赋值
    points是目标
    :return:
    """
    mask = np.zeros(mask_shape)
    for T in triangles:
        mat = np.zeros(mask_shape)
        pos = np.array([np.array(points[T[0]]),
                        np.array(points[T[1]]),
                        np.array(points[T[2]])], dtype=np.int32)
        cv2.fillPoly(mat, [pos], (1, 1, 1))
        mask = (mat - mask) * mat + mask

    # mask = cv2.blur(mask, (mask_shape[0]/5, mask_shape[0]/5))
    return mask


def combine_image(images, mask, shape):
    """
    将变形后的内容通过mask组合
    :return:
    """
    one = np.ones(shape)
    result = np.zeros(shape)
    M = np.zeros(shape)  # 添加这个M是为了防止mask产生重叠
    for i, m in zip(images, mask):
        result = result + i * m
        M = m + M + 0.0001
        one = one - one * m
    return result / M, one


def from_lanmark_to_points(landmarks):
    """
    将图片的标准点格式转换成points格式
    :return:
    """
    points = dict()
    for k in landmarks:
        points[k] = (landmarks[k]["x"], landmarks[k]["y"])
    return points


def draw_triangle(image, triangles, points):
    """
    给出人脸图像、特征点points以及三角剖分结果，将剖分结果绘制在图像上
    :return:
    """
    for T in triangles:
        point = np.array([np.array(points[T[0]]),
                        np.array(points[T[1]]),
                        np.array(points[T[2]])], dtype=np.int32)
        cv2.polylines(image, [point], True, (0, 255, 0))


def trianglulation(points_dict):
    """
    将点三角化，输出三角化的结果
    :return:
    """
    points_list = points_dict.values()
    points_keys = points_dict.keys()

    tri = Delaunay(points_list)
    triangles = []
    for T in tri.simplices:
        tmp = (points_keys[T[0]], points_keys[T[1]], points_keys[T[2]])
        triangles.append(tmp)

    return triangles


def add_landmark(landmark):
    """
    补充landmark
    :param landmark:
    :return:
    """
    landmark["forehead_left_quarter"] = dict()
    landmark["forehead_right_quarter"] = dict()
    landmark["forehead_left_quarter"]['x'] = landmark["left_eyebrow_upper_middle"]['x']
    landmark["forehead_left_quarter"]['y'] = landmark["left_eyebrow_upper_middle"]['y'] + 2 * (landmark["left_eyebrow_upper_middle"]['y'] - landmark["left_eyebrow_lower_middle"]['y'])
    landmark["forehead_right_quarter"]['x'] = landmark["right_eyebrow_upper_middle"]['x']
    landmark["forehead_right_quarter"]['y'] = landmark["right_eyebrow_upper_middle"]['y'] + 2 * (landmark["right_eyebrow_upper_middle"]['y'] - landmark["right_eyebrow_lower_middle"]['y'])

    landmark["forehead_left"] = dict()
    landmark["forehead_right"] = dict()
    landmark["forehead_left"]['x'] = landmark["left_eyebrow_left_corner"]['x']
    landmark["forehead_left"]['y'] = landmark["left_eyebrow_left_corner"]['y'] + 2 * (landmark["left_eyebrow_upper_middle"]['y'] - landmark["left_eyebrow_lower_middle"]['y'])
    landmark["forehead_right"]['x'] = landmark["right_eyebrow_right_corner"]['x']
    landmark["forehead_right"]['y'] = landmark["right_eyebrow_right_corner"]['y'] + 2 * (landmark["right_eyebrow_upper_middle"]['y'] - landmark["right_eyebrow_lower_middle"]['y'])


def add_border_point(landmark, shape):
    """
    添加边界点
    :param landmark:
    :return:
    """
    landmark["top_left"] = {'x': 0, 'y': 0}
    landmark["top_mid"] = {'x': shape[0]/2, 'y': 0}
    landmark["top_right"] = {'x': shape[0], 'y': 0}
    landmark["mid_left"] = {'x': 0, 'y': shape[1] / 2}
    landmark["mid_right"] = {'x': shape[0], 'y': shape[1] / 2}
    landmark["bottom_left"] = {'x': 0, 'y': shape[1]}
    landmark["bottom_mid"] = {'x': shape[0] / 2, 'y': shape[1]}
    landmark["bottom_right"] = {'x': shape[0], 'y': shape[1]}


def get_masked_face(mat, shape):
    landmark = fapi.get_feature_points_fromimage(mat)
    # 这里为了能够更好的提取人脸，手动添加两个特征点

    if landmark is None:
        return None

    add_landmark(landmark)
    '''
    这部分查看一下人脸的特征变形，这里需要计算人脸剖分的方案，需要覆盖全图
    '''
    point_dict = from_lanmark_to_points(landmark)
    triangles = trianglulation(point_dict)

    mask = draw_mask(triangles, point_dict, mat.shape)
    img = np.array(mask * mat, dtype=np.uint8)

    #img[0] = cv2.equalizeHist(img[0])
    #img[1] = cv2.equalizeHist(img[1])
    #img[2] = cv2.equalizeHist(img[2])
    result = img[landmark["forehead_left"]['y']:landmark["contour_chin"]['y'],landmark["contour_left1"]['x']:landmark["contour_right1"]['x']]
    return cv2.resize(result, shape)


def test_face_trianglulation():
    mat = cv2.imread("2.jpg")
    mat = cv2.resize(mat, (mat.shape[0] * 5, mat.shape[1] * 5))
    landmark = fapi.get_feature_points_fromimage(mat)
    # 这里为了能够更好的提取人脸，手动添加两个特征点

    if landmark is None:
        print "人脸识别出错"
    else:
        print "调用face++ api成功"

    add_landmark(landmark)
    add_border_point(landmark, mat.shape)
    '''
    这部分查看一下人脸的特征变形，这里需要计算人脸剖分的方案，需要覆盖全图
    '''
    point_dict = from_lanmark_to_points(landmark)
    triangles = trianglulation(point_dict)
    draw_triangle(mat, triangles, point_dict)

    mask = draw_mask(triangles, point_dict, mat.shape)
    img = np.array(mask * mat, dtype=np.uint8)
    img[0] = cv2.equalizeHist(img[0])
    img[1] = cv2.equalizeHist(img[1])
    img[2] = cv2.equalizeHist(img[2])
    cv2.imshow("mask", img)
    cv2.imshow("mat", mat)
    cv2.waitKey(0)


def test_face_deform():
    """
        对已知剖分结果的三角面进行变形
    """

    src = cv2.imread("tmp_data/NE1.jpg")
    src = cv2.resize(src, (src.shape[1]/3, src.shape[0]/3))

    dst = cv2.imread("tmp_data/SU2_S1.jpg")
    dst = cv2.resize(dst, (src.shape[1], src.shape[0]))

    print src.shape
    print dst.shape
    src_landmark = fapi.get_feature_points_fromimage(src)
    dst_landmark = fapi.get_feature_points_fromimage(dst)
    if dst_landmark is None or src_landmark is None:
        print "人脸识别出错"
    else:
        print "特征点识别成功"

    add_landmark(src_landmark)
    add_landmark(dst_landmark)

    shape = src.shape
    add_border_point(src_landmark, (shape[1],shape[0]))
    add_border_point(dst_landmark, (shape[1],shape[0]))

    '''
    这部分查看一下人脸的特征变形，这里需要计算人脸剖分的方案，需要覆盖全图
    '''
    src_points = from_lanmark_to_points(src_landmark)
    dst_points = from_lanmark_to_points(dst_landmark)
    triangles = trianglulation(src_points)

    """
    draw_triangle(src, triangles, src_points)
    draw_triangle(dst, triangles, dst_points)

    cv2.imshow("src",src)
    cv2.imshow("dst",dst)
    cv2.waitKey()
    """

    affine_mats = compute_affine_mat(triangles, src_points, dst_points)

    warped_images = []
    for M in affine_mats:
        I = cv2.warpAffine(src, M, (shape[1],shape[0]))
        warped_images.append(I)

    masks = draw_masks(triangles, dst_points, shape)
    result, one = combine_image(warped_images, masks, shape)
    result = result + one * src
    # draw_triangle(mat, triangles, points_dst)
    cv2.imshow("dst", result / 255)

    cv2.imshow("src", src)
    cv2.waitKey(0)


def test_fer2013_data():
    train_image, train_label, test_image, test_label = fdata.load_kaggle_face_data()

    for i, mat in enumerate(train_image):
        cv2.imshow("mat", mat)
        cv2.waitKey(0)

        landmark = fapi.get_feature_points_fromimage(mat)

        if landmark is None:
            print "人脸识别出错"
        else:
            print "调用face++ api成功"

        add_landmark(landmark)
        '''
        这部分查看一下人脸的特征变形，这里需要计算人脸剖分的方案，需要覆盖全图
        '''
        point_dict = from_lanmark_to_points(landmark)
        triangles = trianglulation(point_dict)
        # draw_triangle(mat, triangles, point_dict)

        mask = draw_mask(triangles, point_dict, mat.shape)
        img = np.array(mask * mat, dtype=np.uint8)
        img = cv2.equalizeHist(img)
        cv2.imshow("mask", img)
        cv2.imshow("mat", mat)
        cv2.waitKey(0)


if "__main__" == __name__:
    #  test_face_trianglulation()
    test_face_deform()