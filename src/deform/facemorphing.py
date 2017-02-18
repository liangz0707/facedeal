# coding:utf-8
import numpy as np
import cv2
import os
import src.face.faceppapi as fapi


def triangle_face():
    """
    将人脸的特征点划分成三角点
    :return:
    """
    pass


def warp_transform():
    """
    将三角点当中的数据进行变形
    :return:
    """
    pass


def draw_feature_points(face, triangles, feature_points):
    for T in triangles:
        cv2.circle(face, (feature_points[T[0]]['x'], feature_points[T[0]]['y']), 1, (0, 255, 0), -1)
        cv2.circle(face, (feature_points[T[1]]['x'], feature_points[T[1]]['y']), 1, (0, 255, 0), -1)
        cv2.circle(face, (feature_points[T[2]]['x'], feature_points[T[2]]['y']), 1, (0, 255, 0), -1)


def get_affine_mat(triangles, feature_points_src, feature_points_dst):
    """
    计算两个图像对齐的矩阵
    :param triangles:
    :param feature_points_src:
    :param feature_points_dst:
    :return:
    """
    affines = []
    for T in triangles:
        src = np.array([np.array((feature_points_src[T[0]]['x'], feature_points_src[T[0]]['y']), dtype=np.float32),
                             np.array((feature_points_src[T[1]]['x'], feature_points_src[T[1]]['y']), dtype=np.float32),
                             np.array((feature_points_src[T[2]]['x'], feature_points_src[T[2]]['y']), dtype=np.float32)], dtype=np.float32)

        dst = np.array([np.array((feature_points_dst[T[0]]['x'], feature_points_dst[T[0]]['y']), dtype=np.float32),
                             np.array((feature_points_dst[T[1]]['x'], feature_points_dst[T[1]]['y']), dtype=np.float32),
                             np.array((feature_points_dst[T[2]]['x'], feature_points_dst[T[2]]['y']), dtype=np.float32)], dtype=np.float32)

        affines.append(cv2.getAffineTransform(src, dst))
    return affines


def get_aligen_mat(triangles, feature_points):
    """
    把图像重的点和某几个固定坐标对齐
    :param triangles:
    :param feature_points:
    :return:
    """
    affines = []
    for T in triangles:
        src = np.array([np.array((feature_points_src[T[0]]['x'], feature_points_src[T[0]]['y']), dtype=np.float32),
                             np.array((feature_points_src[T[1]]['x'], feature_points_src[T[1]]['y']), dtype=np.float32),
                             np.array((feature_points_src[T[2]]['x'], feature_points_src[T[2]]['y']), dtype=np.float32)], dtype=np.float32)

        dst = np.array([np.array((5, 30), dtype=np.float32),
                             np.array((75, 30), dtype=np.float32),
                             np.array((40, 90), dtype=np.float32)], dtype=np.float32)

        affines.append(cv2.getAffineTransform(src, dst))
    return affines

if "__main__" == __name__:
    """
    {u'mouth_upper_lip_left_contour2': {u'y': 62, u'x': 22},
    u'contour_chin': {u'y': 93, u'x': 39},
    u'mouth_lower_lip_right_contour3': {u'y': 70, u'x': 35},
    u'mouth_upper_lip_left_contour1': {u'y': 59, u'x': 23},
    u'left_eye_upper_left_quarter': {u'y': 31, u'x': 17},
    u'left_eyebrow_lower_middle': {u'y': 25, u'x': 16},
     u'mouth_upper_lip_left_contour3': {u'y': 64, u'x': 25},
      u'left_eyebrow_lower_left_quarter': {u'y': 25, u'x': 13},
      u'nose_contour_left3': {u'y': 51, u'x': 24},
      u'right_eye_pupil': {u'y': 25, u'x': 49},
      u'left_eyebrow_upper_left_quarter': {u'y': 23, u'x': 12},
      u'mouth_lower_lip_left_contour2': {u'y': 69, u'x': 24},
      u'left_eye_bottom': {u'y': 33, u'x': 20},
      u'mouth_lower_lip_bottom': {u'y': 72, u'x': 30},
      u'contour_left9': {u'y': 89, u'x': 32},
      u'left_eye_lower_right_quarter': {u'y': 32, u'x': 22},
       u'mouth_lower_lip_top': {u'y': 67, u'x': 29},
       u'contour_right6': {u'y': 73, u'x': 78},
        u'right_eye_bottom': {u'y': 27, u'x': 48},
         u'contour_right9': {u'y': 92, u'x': 50},
         u'contour_left6': {u'y': 69, u'x': 19},
          u'contour_left5': {u'y': 62, u'x': 16},
          u'contour_left4': {u'y': 54, u'x': 13}, u'contour_left3': {u'y': 47, u'x': 12}, u'contour_left2': {u'y': 38, u'x': 12},
      u'contour_left1': {u'y': 30, u'x': 11},
      u'left_eye_lower_left_quarter': {u'y': 33, u'x': 18},
      u'contour_right1': {u'y': 22, u'x': 83}, u'contour_right3': {u'y': 43, u'x': 87}, u'contour_right2': {u'y': 33, u'x': 86}, u'contour_right5': {u'y': 64, u'x': 83}, u'contour_right4': {u'y': 54, u'x': 86}, u'contour_right7': {u'y': 81, u'x': 70}, u'left_eyebrow_left_corner': {u'y': 26, u'x': 11}, u'nose_right': {u'y': 46, u'x': 42}, u'nose_tip': {u'y': 44, u'x': 23}, u'nose_contour_lower_middle': {u'y': 51, u'x': 28}, u'right_eye_top': {u'y': 24, u'x': 48}, u'mouth_lower_lip_left_contour3': {u'y': 72, u'x': 26}, u'right_eye_right_corner': {u'y': 25, u'x': 57}, u'right_eye_lower_right_quarter': {u'y': 26, u'x': 53}, u'mouth_upper_lip_right_contour2': {u'y': 61, u'x': 38}, u'right_eyebrow_lower_right_quarter': {u'y': 17, u'x': 52}, u'contour_left7': {u'y': 76, u'x': 22}, u'mouth_right_corner': {u'y': 64, u'x': 45}, u'mouth_lower_lip_right_contour1': {u'y': 65, u'x': 37}, u'contour_right8': {u'y': 87, u'x': 61}, u'left_eyebrow_right_corner': {u'y': 24, u'x': 24}, u'right_eye_center': {u'y': 25, u'x': 48}, u'left_eye_pupil': {u'y': 30, u'x': 21}, u'left_eye_upper_right_quarter': {u'y': 29, u'x': 22}, u'mouth_upper_lip_top': {u'y': 59, u'x': 26}, u'nose_left': {u'y': 48, u'x': 20}, u'right_eyebrow_lower_middle': {u'y': 18, u'x': 46}, u'left_eye_top': {u'y': 29, u'x': 19}, u'left_eye_center': {u'y': 31, u'x': 19}, u'contour_left8': {u'y': 83, u'x': 27}, u'right_eyebrow_left_corner': {u'y': 22, u'x': 34}, u'right_eye_left_corner': {u'y': 27, u'x': 41}, u'right_eyebrow_lower_left_quarter': {u'y': 20, u'x': 40}, u'left_eye_left_corner': {u'y': 33, u'x': 15}, u'mouth_left_corner': {u'y': 66, u'x': 24}, u'right_eyebrow_upper_left_quarter': {u'y': 16, u'x': 39}, u'left_eye_right_corner': {u'y': 30, u'x': 25}, u'right_eye_lower_left_quarter': {u'y': 26, u'x': 45}, u'right_eyebrow_right_corner': {u'y': 18, u'x': 59}, u'right_eye_upper_left_quarter': {u'y': 24, u'x': 45}, u'left_eyebrow_upper_middle': {u'y': 21, u'x': 16}, u'mouth_lower_lip_right_contour2': {u'y': 67, u'x': 41}, u'nose_contour_left1': {u'y': 30, u'x': 27}, u'nose_contour_left2': {u'y': 41, u'x': 21}, u'mouth_upper_lip_right_contour1': {u'y': 59, u'x': 29}, u'nose_contour_right1': {u'y': 29, u'x': 33}, u'nose_contour_right2': {u'y': 39, u'x': 37}, u'nose_contour_right3': {u'y': 50, u'x': 35}, u'mouth_upper_lip_bottom': {u'y': 63, u'x': 27}, u'right_eyebrow_upper_middle': {u'y': 14, u'x': 45}, u'left_eyebrow_lower_right_quarter': {u'y': 24, u'x': 20}, u'right_eyebrow_upper_right_quarter': {u'y': 14, u'x': 52}, u'mouth_upper_lip_right_contour3': {u'y': 62, u'x': 36}, u'left_eyebrow_upper_right_quarter': {u'y': 22, u'x': 20}, u'right_eye_upper_right_quarter': {u'y': 24, u'x': 52}, u'mouth_lower_lip_left_contour1': {u'y': 67, u'x': 26}}
    """
    #Ts = [["left_eyebrow_right_corner", "nose_tip", "right_eyebrow_left_corner"]]
    Ts = [["left_eyebrow_lower_right_quarter", "left_eyebrow_upper_middle", "left_eyebrow_upper_right_quarter"]]
    aligenT = [["contour_left1", "contour_right1", "contour_chin"]]

    # 获取头像和头像的特征点
    dir = "../../dataset/expression_rect/"
    # face = fapi.from_filename2face(dir+"SA1003.jpg", (80, 80))  # SU1010.jpg
    face_src = cv2.imread(dir + "FE394.jpg")
    face_dst = cv2.imread(dir + "SA1010.jpg")

    face_src[:,:,0]= cv2.equalizeHist(face_src[:,:,0])
    face_dst[:,:,0] = cv2.equalizeHist(face_dst[:,:,0])
    face_src[:,:,1]= cv2.equalizeHist(face_src[:,:,1])
    face_dst[:,:,1] = cv2.equalizeHist(face_dst[:,:,1])
    face_src[:,:,2]= cv2.equalizeHist(face_src[:,:,2])
    face_dst[:,:,2] = cv2.equalizeHist(face_dst[:,:,2])

    feature_points_src = fapi.get_feature_points_fromimage(face_src, verbose=True)
    draw_feature_points(face_src, Ts, feature_points_src)

    feature_points_dst = fapi.get_feature_points_fromimage(face_dst, verbose=True)
    draw_feature_points(face_dst, Ts, feature_points_dst)

    aligens_src = get_aligen_mat(aligenT, feature_points_src)
    aligens_dst = get_aligen_mat(aligenT, feature_points_dst)

    al_src = cv2.warpAffine(face_src, aligens_src[0], face_dst.shape[0:2])
    al_dst = cv2.warpAffine(face_dst, aligens_dst[0], face_dst.shape[0:2])

    conva = np.zeros((face_src.shape[0], face_src.shape[1] * 3, face_src.shape[2]))

    conva[:, 0:face_src.shape[1], :] = face_src
    conva[:, face_src.shape[1]:face_src.shape[1] + face_src.shape[1], :] = al_src
    conva[:, 2 * face_src.shape[1]:, :] = al_dst
    print conva.shape
    print face_src.shape
    cv2.imshow("face", conva/255)
    cv2.waitKey(0)

