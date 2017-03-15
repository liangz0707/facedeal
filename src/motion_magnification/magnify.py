# coding:utf-8
import cv2
import numpy as np
import os


class Magnify:
    def __init__(self):
        self.rate = 0
        self.fnumber = 0
        self.lowpass1 = None
        self.lowpass2 = None
        self.levels = 8
        self.spatialType = "L"
        self.temporalType = "C"
        self.exaggeration_factor = 2.0
        self.alpha = 80.0  # 放大因子
        self.lamb = 0
        # cut-off wave length
        self.lambda_c = 40.0
        # low cut-off
        self.fl = 0.23
        # high 35ut-off
        self.fh = .40
        self.curLevel = 0
        self.delta = 1.0
        self.chromAttenuation = 0.1  # 颜色衰减

    def build_laplacian_pyramid(self, img, levels):
        pyramid = []
        if levels < 1:
            print("Levels should be larger than 1")
            return False, None
        current_img = np.copy(img)

        for i in range(levels):
            down = cv2.pyrDown(current_img)
            up = cv2.pyrUp(down)
            up = cv2.resize(up, (current_img.shape[1], current_img.shape[0]))
            lap = np.add(current_img, -1.0 * up)
            pyramid.append(lap)
            current_img = down

        pyramid.append(current_img)
        return True, pyramid

    def build_gauss_pyramid(self, img, levels):
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

    def reconstruct_image_from_laplacian_pyramid(self, pyramid):
        levels = len(pyramid) - 1
        current = pyramid[levels]
        for i in range(levels - 1, -1, -1):
            up = cv2.pyrUp(current)
            current = np.add(up, pyramid[i])
        return current

    def upsampling_fromGaussian_pyramid(self, src, levels):
        current_level = np.array(src)
        for i in range(levels):
            up = None
            cv2.pyrUp(current_level, up)
            current_level = up
        return current_level

    def temporal_IIR_filter(self, src):
        temp1 = (1-self.fh)*self.lowpass1[self.curLevel] + self.fh*src
        temp2 = (1-self.fl)*self.lowpass2[self.curLevel] + self.fl*src
        self.lowpass1[self.curLevel] = temp1
        self.lowpass2[self.curLevel] = temp2
        dst = self.lowpass1[self.curLevel] - self.lowpass2[self.curLevel]
        return dst

    def temporal_ideal_filter(self, src):
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
            filter = self.create_ideal_bandpass_filter(filter, self.fl, self.fh, self.rate)
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


    def create_ideal_bandpass_filter(self, filter, fl, fh, rate):
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

    def setSpatialFilter(self, F):
        spatialType = F

    def setTemporalFilter(self, F):
        temporalType = F


    def spatialFilter(self, input, levels):
        if self.spatialType == "L":
            return self.build_laplacian_pyramid(input, levels)
        else:
            return self.build_gauss_pyramid(input, levels)


    def temporalFilter(self, src):
        if self.temporalType == "I":
            return self.temporal_ideal_filter(src)
        else:
            return self.temporal_IIR_filter(src)


    def amplify(self, src):
        if self.spatialType == "L":
            # compute modified alpha for this level
            currAlpha = self.lamb/ self.delta/8.0 - 1
            currAlpha = self.exaggeration_factor * currAlpha
            if self.curLevel == self.levels or self.curLevel == 0:    # ignore the highest and lowest frequency band
                dst = src * 0
            else:
                dst = src * np.min((self.alpha, currAlpha))
        else:
            dst = src * self.alpha

        return dst

    def attenuate(self, src):
        dst = np.copy(src)
        planes = cv2.split(src)

        planes[1] = np.copy(planes[1] * self.chromAttenuation)
        planes[2] = np.copy(planes[2] * self.chromAttenuation)
        cv2.merge(planes, dst)
        return dst

    def magnify_video(self, input_file, output_file, levels=8):

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
        self.rate = fps  # 获取帧率

        while vc.isOpened():
            ret, input = vc.read()

            if not ret:
                break

            input = cv2.resize(input, size)
            input = np.float32(input / 255.0)  # ?

            back = np.copy(input)

            input = cv2.cvtColor(input, cv2.COLOR_RGB2LAB)

            s = np.copy(input)
            ret, pyramid = self.spatialFilter(input, levels)
            filtered = pyramid[:]

            if self.fnumber == 0:
                self.lowpass1 = pyramid[:]
                self.lowpass2 = pyramid[:]
                filtered = pyramid[:]
            else:

                for i in range(levels):
                    self.curLevel = i
                    filtered[i] = self.temporalFilter(pyramid[i])

                h, w, _ = input.shape
                self.delta = self.lambda_c / 8.0 / (1.0 + self.alpha)
                self.lamb = np.power(w * w + h * h, 0.5) / 3.0  # 3 is experimental constant
                self.exaggeration_factor = 2.0

                for i in range(self.levels, -1, -1):
                    self.curLevel = i

                    filtered[i] = self.amplify(filtered[i])

                    # go one level down on pyramid
                    # representative lambda will reduce by factor of 2
                    self.lamb = self.lamb / 2.0

            motion = self.reconstruct_image_from_laplacian_pyramid(filtered)
            motion = self.attenuate(motion)

            if self.fnumber > 0:  # don't amplify first frame
                s = np.add(s, motion)

            output = np.float32(np.copy(s))
            output = cv2.cvtColor(output, cv2.COLOR_LAB2RGB)
            self.fnumber = 1 + self.fnumber
            print self.fnumber

            comp = np.zeros((size[1], size[0] * 2, 3), dtype=np.uint8)

            comp[:, 0: size[0], :] = np.uint8(back * 255)
            comp[:, size[0]:, :] = np.uint8(output * 255)
            vw.write(comp)

        vw.release()
        vc.release()


    def magnify_frame(self, img1, img2, levels =8):
        self.levels = levels
        fps = 30
        n = np.power(2, levels)
        size = (img1.shape[1], img1.shape[0])

        size = (size[0] / n * n, size[1] / n * n)
        self.rate = fps  # 获取帧率

        input = np.copy(img1)
        input = cv2.resize(input, size)
        input = np.float32(input / 255.0)
        input = cv2.cvtColor(input, cv2.COLOR_RGB2LAB)
        ret, pyramid = self.spatialFilter(input, levels)
        self.lowpass1 = pyramid[:]
        self.lowpass2 = pyramid[:]

        input = np.copy(img2)
        input = cv2.resize(input, size)
        input = np.float32(input / 255.0)
        back = np.copy(input)
        input = cv2.cvtColor(input, cv2.COLOR_RGB2LAB)
        s = np.copy(input)
        ret, pyramid = self.spatialFilter(input, levels)
        filtered = pyramid[:]

        for i in range(levels):
            self.curLevel = i
            filtered[i] = self.temporalFilter(pyramid[i])

        h, w, _ = input.shape
        self.delta = self.lambda_c / 8.0 / (1.0 + self.alpha)
        self.lamb = np.power(w * w + h * h, 0.5) / 3.0  # 3 is experimental constant
        self.exaggeration_factor = 2.0

        for i in range(self.levels, -1, -1):
            self.curLevel = i

            filtered[i] = self.amplify(filtered[i])
            self.lamb = self.lamb / 2.0

        motion = self.reconstruct_image_from_laplacian_pyramid(filtered)
        motion = self.attenuate(motion)

        s = np.add(s, motion)

        output = np.float32(np.copy(s))
        output = cv2.cvtColor(output, cv2.COLOR_LAB2RGB)

        comp = np.zeros((size[1], size[0] * 2, 3), dtype=np.uint8)

        comp[:, 0: size[0], :] = np.uint8(back * 255)
        comp[:, size[0]:, :] = np.uint8(output * 255)
        return np.uint8(output * 255), comp


if "__main__" == __name__:
    mg = Magnify()
    img1 = cv2.imread("../../dataset/EP02_01f/img46.jpg")
    img2 = cv2.imread("../../dataset/EP02_01f/img59.jpg")
    #img1 = cv2.resize(img1, (0,0), None , 2,2)
    #img2 = cv2.resize(img2, (0,0), None , 2,2)
    r,c = mg.magnify_frame(img1, img2)
    r,c = mg.magnify_frame(img2, r)
    cv2.imshow("a", c)
    cv2.waitKey(0)
    #mg.magnify_video("../../dataset/video/EP02_01f.mov", "../../dataset/EP.avi", 8)


