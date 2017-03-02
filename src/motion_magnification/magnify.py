# coding:utf-8
import cv2
import numpy as np
import os


rate = 0
fnumber = 0
lowpass1 = None
lowpass2 = None
levels = 8
spatialType = "L"
temporalType = "C"
exaggeration_factor = 2.0
alpha = 10.0  # 放大因子
lamb = 0
# cut-off wave length
lambda_c = 15.0
# low cut-off
fl = 0.3
# high 35ut-off
fh = 0.9
curLevel = 0
delta = 1.0
chromAttenuation = 0.2  # 颜色衰减


def build_laplacian_pyramid(img, levels):
    pyramid = []
    if levels < 1:
        print("Levels should be larger than 1")
        return False, None
    current_img = np.copy(img)

    for i in range(levels):
        down = cv2.pyrDown(current_img)
        up = cv2.pyrUp(down)
        up = cv2.resize(up, (current_img.shape[1], current_img.shape[0]))
        lap = np.add(current_img, -1.0*up)
        pyramid.append(lap)
        current_img = down

    pyramid.append(current_img)
    return True, pyramid


def build_gauss_pyramid(img, levels):
    pyramid = []
    if levels < 1:
        print("Levels should be larger than 1")
        return False, None

    current_img = np.array(img)
    for i in range(levels):
        down = cv2.pyrDown(current_img)
        pyramid.append(down)
        current_img = down

    pyramid.append(current_img)
    return True, pyramid


def reconstruct_image_from_laplacian_pyramid(pyramid):
    levels = len(pyramid) - 1
    current = pyramid[levels]
    for i in range(levels - 1, -1, -1):
        up = cv2.pyrUp(current)
        current = np.add(up, pyramid[i])
    return current


def upsampling_fromGaussian_pyramid(src, levels):
    current_level = np.array(src)
    for i in range(levels):
        up = None
        cv2.pyrUp(current_level, up)
        current_level = up
    return current_level


def temporal_IIR_filter(src):
    temp1 = (1-fh)*lowpass1[curLevel] + fh*src
    temp2 = (1-fl)*lowpass2[curLevel] + fl*src
    lowpass1[curLevel] = temp1
    lowpass2[curLevel] = temp2
    dst = lowpass1[curLevel] - lowpass2[curLevel]
    return dst


def temporal_ideal_filter(src):
    channels = cv2.split(src)
    for i in range(len(channels)):
        current = channels[i]
        width = cv2.getOptimalDFTSize(current.shape[1])
        height = cv2.getOptimalDFTSize(current.shape[0])
        temp_img = cv2.copyMakeBorder(current,
                           0, height - current.shape[0],
                           0, width - current.shape[1],
                           cv2.BORDER_CONSTANT, (0,0,0,0))
        # do the DFT
        cv2.dft(temp_img, temp_img, cv2.DFT_ROWS | cv2.DFT_SCALE, temp_img.shape[0])
        # construct the filter
        filter = np.copy(temp_img)
        filter = create_ideal_bandpass_filter(filter, fl, fh, rate)
        # apply filter
        temp_img = cv2.mulSpectrums(temp_img, filter, cv2.DFT_ROWS)
        # do the inverse DFT on filtered image
        cv2.idft(temp_img, temp_img, cv2.DFT_ROWS | cv2.DFT_SCALE, temp_img.shape[0])
        # copy back to the current channel
        channels[i] = temp_img[0:current.shape[0], 0:current.shape[1]]

    # merge channels]
    dst = cv2.merge(channels)
    # normalize the filtered image
    dst = cv2.normalize(dst, None, 0, 1, cv2.NORM_MINMAX)
    return dst


def create_ideal_bandpass_filter(filter, fl, fh, rate):
    height, width = filter.shape
    fl = 2 * fl * width / rate
    fh = 2 * fh * width / rate
    for i in range(height):
        for j in range(width):
            # filter response
            if j >= fl and j <= fh:
                response = 1.0
            else:
                response = 0.0
            filter[i, j] = response
    return filter


def setSpatialFilter(F):
    spatialType = F


def setTemporalFilter(F):
    temporalType = F


def spatialFilter(input, levels):
    if spatialType == "L":
        return build_laplacian_pyramid(input, levels)
    else:
        return build_gauss_pyramid(input, levels)


def temporalFilter(src):
    if temporalType == "I":
        return temporal_ideal_filter(src)
    else:
        return temporal_IIR_filter(src)


def amplify(src):
    if spatialType == "L":
        # compute modified alpha for this level
        currAlpha = lamb/delta/8.0 - 1
        currAlpha = exaggeration_factor * currAlpha
        if curLevel == levels or curLevel == 0:    # ignore the highest and lowest frequency band
            dst = src * 0
        else:
            dst = src * np.min((alpha, currAlpha))
    else:
        dst = src * alpha

    return dst


def attenuate(src):
    dst = np.copy(src)
    planes = cv2.split(src)

    planes[1] = np.copy(planes[1] * chromAttenuation)
    planes[2] = np.copy(planes[2] * chromAttenuation)
    cv2.merge(planes, dst)
    return dst


if "__main__" == __name__:
    input_file = "../../dataset/video/face.mov"

    output_file= "../../dataset/video/face.avi"

    if os.path.exists(output_file):
        os.remove(output_file)

    vc = cv2.VideoCapture(input_file)
    if vc.isOpened():
        print "视频成功打开！"
    else:
        print "video can't open"

    fps = vc.get(cv2.cv.CV_CAP_PROP_FPS)

    n = np.power(2, levels)
    size = (int(vc.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
            int(vc.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))

    size = (size[0] / n * n, size[1] / n * n)
    size_wh = (size[0] / n * n * 2, size[1] / n * n)

    vw = cv2.VideoWriter(output_file, -1, fps, size_wh)
    if vw.isOpened():
        print "视频打成功"
    rate = fps  # 获取帧率

    while vc.isOpened():
        ret, input = vc.read()

        if not ret:
            break

        input = cv2.resize(input, size)
        input = np.float32(input / 255.0)  # ?

        back = np.copy(input)

        input = cv2.cvtColor(input, cv2.COLOR_RGB2LAB)

        s = np.copy(input)
        ret, pyramid = spatialFilter(input, levels)
        filtered = pyramid[:]

        if fnumber == 0:
            lowpass1 = pyramid[:]
            lowpass2 = pyramid[:]
            filtered = pyramid[:]
        else:

            for i in range(levels):
                curLevel = i
                filtered[i] = temporalFilter(pyramid[i])

            h, w, _ = input.shape
            delta = lambda_c / 8.0 / (1.0 + alpha)
            lamb = np.power(w*w + h*h, 0.5)/3.0  # 3 is experimental constant
            exaggeration_factor = 2.0

            for i in range(levels, -1, -1):
                curLevel = i

                filtered[i] = amplify(filtered[i])

                # go one level down on pyramid
                # representative lambda will reduce by factor of 2
                lamb = lamb / 2.0

        motion = reconstruct_image_from_laplacian_pyramid(filtered)
        motion = attenuate(motion)

        if (fnumber > 0): # don't amplify first frame
            s = np.add(s, motion)

        output = np.float32(np.copy(s))
        output = cv2.cvtColor(output, cv2.COLOR_LAB2RGB)
        fnumber = 1 + fnumber
        print fnumber

        comp = np.zeros((size[1], size[0] * 2, 3), dtype=np.uint8)

        comp[:, 0: size[0], :] = np.uint8(back * 255)
        comp[:, size[0]:, :] = np.uint8(output * 255)
        vw.write(comp)

    vw.release()
    vc.release()
    cv2.destroyAllWindows()