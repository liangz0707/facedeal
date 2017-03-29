# coding:utf-8

import matplotlib
matplotlib.use('TkAgg')

import data_collector.data_input as din
import expression_recognize.facevisualize as fv

import cv2
import matplotlib.pyplot as plt
import cPickle
import numpy as np
import img_toolkits.OpticalFlow_feature as of
import expression_recognize.facedetection as fd
import os
import time

def face_vis():
    train_image, train_label, test_image, test_image = din.load_kaggle_face_data(max_len=1000)

    Mat, evals, evecs = fv.feature_face(train_image, dim=5)

    print len(train_image)
    print Mat.shape
    print evals.shape
    print evecs.shape

    plt.figure(1)

    for i in range(len(train_image)):
        plt.plot(Mat[:, 0], Mat[:, 1], "b.")

        plt.plot(Mat[i, 0], Mat[i, 1], "r.")

        cv2.imshow("ex", train_image[i].reshape(48, 48))
        plt.show()


def ck_vis():
    '''
    人脸的样子进行PCA降维后，最开始 是全脸的样子，往后，就出现了对人的五官的权重的加深
    :return:
    '''
    ck_file = open("ck_faces_miniversion.cp", "rb")
    train_image, train_label, test_label = cPickle.load(ck_file)
    #train_image = cPickle.load(ck_file)
    Mat, evals, evecs = fv.feature_face(train_image, 10)
    print Mat.shape

    print evecs.shape
    plt.figure(1)

    color_map = ["r.", "g.", "b.", "c.", "m.", "y.", "k.", "r."]
    for i, m in enumerate(evecs.T):
        n = np.reshape(m, (48, 48))
        img = cv2.normalize(n, 0, 1, cv2.NORM_MINMAX)
        cv2.imshow("a", cv2.resize(img*255, (img.shape[0] * 5, img.shape[1]*5)))

        cv2.imshow("b", cv2.resize(train_image[0], (img.shape[0] * 5, img.shape[1]*5)))

        cv2.imshow("c", cv2.resize(train_image[0] * img, (img.shape[0] * 5, img.shape[1]*5)))
        cv2.waitKey(0)
    ''''''
    for i in range(len(Mat)):
        plt.plot(Mat[i, 2], Mat[i, 1], color_map[train_label[i]])

    plt.show()


def ck_training_data_trasform():

    faces, labels, test_faces = din.load_ckplus()
    print len(faces)
    print len(labels)
    print len(test_faces)

    print faces[0].shape

    gray_faces = []
    for face in faces:
        gray_faces.append(cv2.cvtColor(face, cv2.COLOR_RGB2GRAY))

    gray_tests = []
    for face in test_faces:
        gray_tests.append(cv2.cvtColor(face, cv2.COLOR_RGB2GRAY))

    ck_faces = open("ck_faces_full.cp", "wb")

    cPickle.dump((gray_faces, labels, gray_tests), ck_faces)


def camse2_data_transfer() :
    """
    将CAMSE2的数据中的人脸，和特征点全部提取出来
    :return:
    """
    path = "D:\\CAME2\\CASME2-coding-20140508.csv"
    root = "D:\\CAME2\\CASME2_RAW_selected\\"
    rect_root = "D:\\CAME2\\Face_Rect\\"
    land_root = "D:\\CAME2\\Face_Rect\\"
    file = open(path)
    lines = file.readlines()
    for index, line in enumerate(lines[1:]):

        meta = line.split(",")

        subject = "sub%02d" % int(meta[0])
        if os.path.exists(rect_root + subject):
            pass
        else:
            os.mkdir(rect_root + subject)
        sqe = meta[1]

        if os.path.exists(rect_root +subject + "\\" + sqe):
            pass
        else:
            os.mkdir(rect_root + subject + "\\" + sqe)

        begin = int(meta[3])
        end = int(meta[5])
        for i in range(begin, end + 1, 1) :
            file_name = root + subject + "\\" + sqe + "\\" + "img" + str(i) + ".jpg"
            output_name = rect_root + subject + "\\" + sqe + "\\" + "img" + str(i) + ".jpg"
            land_name = land_root + subject + "\\" + sqe + "\\" + "img" + str(i) + ".cp"
            if not os.path.exists(file_name):
                continue
            if os.path.exists(output_name):
                continue
            face = fd.from_filename2face(file_name, (280,320))
            if face is None:
                print "在图片%s中未检测到人脸" % (file_name)
                continue
            cv2.imwrite(output_name,face)

            # file = open(output_name, "rb")
            # landmark,_ = fd.get_feature_point(file)
            # file.close()
            #
            # file = open(land_name, "wb")
            # cPickle.dump(landmark,file)
            # file.close()
            # print landmark


def check_run():
    try :

        camse2_data_transfer()
    except:
        time.sleep(30)
        print "链接出现一次异常"
        check_run()


def camse2_rect_aligened() :
    """
    将CAMSE2的数据中的人脸，和特征点全部提取出来
    :return:
    """
    """`
    人脸对齐：
    1。读取光流，文件中的内容
    2。读取两帧图像
    3。读取第一帧图像的特征点
    4。计算由光流变到第二帧后的位置。
    5。计算仿射变换
    6。对第二张图像进行仿射变换来对齐
    """
    counter_index = range(0,83) # [ 21, 20, 19, 56, 15, 14, 41, 65, 32, 76, 36, 35,44,68 , 60, 30,46]
    path = "D:\\CAME2\\CASME2-coding-20140508.csv"
    rect_root = "D:\\CAME2\\Face_Rect\\"
    flow_root = "D:\\CAME2\\Flow\\"
    aligned_root = "D:\\CAME2\\Face_Rect_Aligned\\"
    file = open(path)
    lines = file.readlines()
    for index, line in enumerate(lines[1:]):

        meta = line.split(",")

        subject = "sub%02d" % int(meta[0])
        if os.path.exists(aligned_root + subject):
            pass
        else:
            os.mkdir(aligned_root + subject)
        sqe = meta[1]

        if os.path.exists(aligned_root +subject + "\\" + sqe):
            pass
        else:
            os.mkdir(aligned_root + subject + "\\" + sqe)

        begin = int(meta[3])
        end = int(meta[5])

        image1_file_name = rect_root + subject + "\\" + sqe + "\\" + "img" + str(begin) + ".jpg"
        image1 = cv2.imread(image1_file_name)
        aligned_name = aligned_root + subject + "\\" + sqe + "\\" + "img" + str(begin) + ".jpg"
        cv2.imwrite(aligned_name, image1[10:-10,10:-10,:])
        vd = fd.get_feature_points_fromimage(image1)
        points = np.array(fd.from_lanmark_to_points(vd).values())

        feature_pos = points[counter_index]

        for i in range(begin + 1, end + 1, 1) :
            flow_name = flow_root + subject + "\\" + sqe + "\\" + "reg_flow" + str(i - 1) + ".xml"
            image2_file_name = rect_root + subject + "\\" + sqe + "\\" + "img" + str(i) + ".jpg"
            image2 = cv2.imread(image2_file_name)
            aligned_name = aligned_root + subject + "\\" + sqe + "\\" + "img" + str(i) + ".jpg"
            if os.path.exists(aligned_name):
                continue
            U, V = flow_reader(flow_name)
            feature_back = []
            for pos in feature_pos:
                pos[0] = max(pos[0], 0)
                pos[1] = max(pos[1], 0)
                pos[0] = min(pos[0], image2.shape[1] - 1)
                pos[1] = min(pos[1], image2.shape[0] - 1)
                new_pos = (pos[0] + U[pos[1]][pos[0]], pos[1] + V[pos[1]][pos[0]])
                feature_back.append(new_pos)
            fl = of.OpticalFlow()
            mat = fl.calc_affine_mat(np.array(feature_back, dtype=np.float32),
                                         np.array(feature_pos, dtype=np.float32))

            B = cv2.warpAffine(image2, mat.T, (image2.shape[1], image2.shape[0]))
            cv2.imwrite(aligned_name, B[10:-10,10:-10,:])

def test_face_aligen() :
    pass
    rect_root = "D:\\CAME2\\Face_Rect\\"
    flow_root = "D:\\CAME2\\Flow\\"
    img_path = "sub01\\EP02_01f\\"
    flow_name = "reg_flow72.xml"
    img_name = "img47.jpg"
    img_name = "img46.jpg"

    U, V = flow_reader(flow_root + img_path + flow_name)
    st = np.power(np.add(V * V ,U * U),0.5)
    deg = np.arctan(V/U)
    cv2.normalize(st,st,0,1,cv2.NORM_MINMAX)

    # 查看旋转角度
    dot = np.histogram(deg,100)
    plt.plot(dot[1][1:],dot[0])
    plt.show()

    # 去除掉明显移动的，和不移动的
    # 将稳定移动的区域进行对其

    stabel_down = st > 0.01
    stabel_up = st < 0.07
    stabel_point = stabel_up & stabel_down
    st[stabel_point ] = 1
    st[~ stabel_point] = 0

    cv2.imshow("a", deg )
    cv2.waitKey(0)

def flow_reader(file_path):
    """
    从FileSortage文件当中读取保存的数据
    :return:
        """
    U = np.asarray(cv2.cv.Load(file_path, cv2.cv.CreateMemStorage(), "U"))
    V = np.asarray(cv2.cv.Load(file_path, cv2.cv.CreateMemStorage(), "V"))
    return U, V


def test_affine_mat():
    feature_point_front = np.array([(1, 2), (3, 4), (3, 8)], dtype=np.float32)
    feature_point_back = feature_point_front * 1
    op = of.OpticalFlow()
    mat = op.calc_affine_mat(feature_point_front, feature_point_back)
    print mat
    pass


if __name__ == '__main__':
    camse2_rect_aligened()



