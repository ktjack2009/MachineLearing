import cv2 as cv
import numpy as np


def create_img():
    # 创建图片
    src = np.zeros((400, 400, 3), dtype=np.uint8)
    src[:, :, 0] = np.ones((400, 400)) * 255
    src[:, :, 2] = np.ones((400, 400)) * 255
    cv.namedWindow('input image', cv.WINDOW_AUTOSIZE)  # 设置图像显示窗口
    cv.imshow('input image', src)


def video_demo():
    # 读取视频
    capture = cv.VideoCapture(0)
    # capture.open(filename=None) # 打开视频文件
    cv.namedWindow('video', cv.WINDOW_AUTOSIZE)
    while True:
        ret, frame = capture.read()
        frame = cv.flip(frame, 1)  # 摄像头默认左右颠倒，code=1镜像，code=0左右上下颠倒，code=-1上下颠倒
        cv.imshow("video", frame)
        c = cv.waitKey(50)  # 延时50ms后返回按键，如果未按，返回-1
        if c == 27:  # esc ascii码的值
            break
        # elif c == 97:
        #     cv.imwrite('/home/1.png', frame)   # 保存图片
    cv.destroyAllWindows()


def pic_demo(path):
    # 图片处理
    src = cv.imread(path)  # 返回一个ndarray对象，加载出来都是BGR
    # src = cv.bitwise_not(src)   # 反色
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)  # 灰度图
    cv.imshow('gray', gray)

    hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
    cv.imshow('hsv', hsv)

    yuv = cv.cvtColor(src, cv.COLOR_BGR2YUV)
    cv.imshow('yuv', yuv)

    Ycrcb = cv.cvtColor(src, cv.COLOR_BGR2YCrCb)
    cv.imshow('Ycrcb', Ycrcb)


def extract_object_demo(path):
    src = cv.imread(path)
    # b, g, r = cv.split(src)  # 通道分离
    # src = cv.merge((b, g, r))  # 通道合并
    hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
    lower_hsv = np.array([35, 43, 46])
    upper_hsv = np.array([77, 255, 255])
    mask = cv.inRange(hsv, lower_hsv, upper_hsv)
    # src[:, :, 0:2].fill(0)  # 通道b、g赋值为0
    cv.imshow('src', src)
    cv.imshow('hsv', hsv)
    cv.imshow('mask', mask)
    # cv.imshow('b', b)
    # cv.imshow('g', g)
    # cv.imshow('r', r)


def calculate_pic():
    def add_demo(m1, m2):
        dst = cv.add(m1, m2)
        cv.imshow('add_demo', dst)

    def sub_demo(m1, m2):
        cv.imshow('sub_demo_1', cv.subtract(m1, m2))
        cv.imshow('sub_demo_2', cv.subtract(m2, m1))

    def divide_demo(m1, m2):
        cv.imshow('divide_demo_1', cv.divide(m1, m2))
        cv.imshow('divide_demo_2', cv.divide(m2, m1))

    def multiply_demo(m1, m2):
        cv.imshow('multiply_demo', cv.multiply(m1, m2))

    def others_demo(m1, m2):
        print(cv.meanStdDev(m1))
        print(cv.meanStdDev(m2))

    src1 = cv.imread('/Users/dsj/Desktop/timg.jpeg')
    src2 = cv.imread('/Users/dsj/Desktop/timg2.jpeg')
    src1 = cv.resize(src1, src2.shape[:2][::-1])  # resize图片
    cv.imshow('src1', src1)
    cv.imshow('src2', src2)
    add_demo(src1, src2)
    sub_demo(src1, src2)
    divide_demo(src1, src2)
    multiply_demo(src1, src2)
    others_demo(src1, src2)
    cv.waitKey(0)
    cv.destroyAllWindows()


def logic_demo():
    src1 = cv.imread('/Users/dsj/Desktop/timg.jpeg')
    src2 = cv.imread('/Users/dsj/Desktop/timg2.jpeg')
    src1 = cv.resize(src1, src2.shape[:2][::-1])  # resize图片
    cv.imshow('logic_and', cv.bitwise_and(src1, src2))
    cv.imshow('logic_or', cv.bitwise_or(src1, src2))
    cv.imshow('logic_xor', cv.bitwise_xor(src1, src2))


def contrast_brightness_demo():
    c = 1  # 对比度
    b = 0  # 亮度
    src = cv.imread('/Users/dsj/Desktop/timg.jpeg')
    bank = np.zeros(src.shape, dtype=src.dtype)
    dst = cv.addWeighted(src, c, bank, 1 - c, b)
    cv.imshow('dst', dst)
    cv.imshow('src', src)


def roi_demo():
    # 通过numpy获取roi
    src = cv.imread('/Users/dsj/Desktop/timg.jpeg')
    cv.imshow('src0', src)
    face = src[90:300, 300:500]
    gray = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
    back_face = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    src[90:300, 300:500] = back_face
    cv.imshow('src', src)


def fill_color_demo():
    # 泛洪填充
    src = cv.imread('/Users/dsj/Desktop/timg.jpeg')
    copyImg = src.copy()
    h, w = src.shape[:2]
    mask = np.zeros([h + 2, w + 2], np.uint8)  # mask必须是这中形式
    # 参数
    # 从(30, 30)像素点开始，填充为(0, 255, 255)【黄色】
    # 高值(100, 100, 100), 低值(50, 50, 50) 越窄越精细
    cv.floodFill(copyImg, mask, (30, 30), (0, 255, 255), (20, 20, 20), (20, 20, 20), cv.FLOODFILL_FIXED_RANGE)
    cv.imshow('src', src)
    cv.imshow('demo', copyImg)


def fill_binary():
    # 二值图像填充
    src = np.zeros([400, 400, 3], np.uint8)
    src[100:300, 100:300, :] = 255
    cv.imshow('src', src)
    mask = np.ones([402, 402, 1], np.uint8)
    mask[101:301, 101:301] = 0
    cv.floodFill(src, mask, (200, 200), (0, 0, 255), cv.FLOODFILL_MASK_ONLY)
    cv.imshow('demo', src)


def blur_demo():
    # 模糊操作
    def custom_blur_demo(image):
        # 自定义模糊
        # kernel = np.ones([5, 5], np.float32) / 25
        kernel = np.array([[0, -1, 0], [0, 3, 0], [0, -1, 0]], np.float32) / 1  # 锐化
        dst = cv.filter2D(image, -1, kernel=kernel)
        cv.imshow('custom', dst)

    src = cv.imread('/Users/dsj/Desktop/timg.jpeg')
    cv.imshow('src', src)

    dst = cv.blur(src, (5, 5))  # 均值模糊，去随机噪声
    median = cv.medianBlur(src, 5)  # 中值模糊，去椒盐噪声
    gaussian = cv.GaussianBlur(src, (3, 3), 4)  # 高斯模糊
    cv.imshow('dst', dst)
    cv.imshow('median', median)
    cv.imshow('gaussian', gaussian)
    custom_blur_demo(src)


def epf():
    # 边缘保留滤波【高斯双边、均值迁移】
    def bi_demo(image):
        # 高斯双边
        dst = cv.bilateralFilter(image, 0, 100, 2)
        cv.imshow('bi_demo', dst)

    def bi_demo2(image):
        # 均值迁移
        dst = cv.pyrMeanShiftFiltering(image, 5, 10)
        cv.imshow('bi_demo2', dst)

    src = cv.imread('/Users/dsj/Desktop/timg.jpeg')
    cv.imshow('src', src)
    bi_demo(src)
    bi_demo2(src)


def binary_image():
    # 图像二值化
    src = cv.imread('/Users/dsj/Desktop/timg.jpeg')

    def threshold_demo():
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)
        # ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_TOZERO)
        # ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_TRIANGLE)
        cv.imshow('binary', binary)

    def local_threshold():
        # 局部二值化
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        # binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 25, 10)
        binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 10)
        cv.imshow('binary', binary)

    local_threshold()


def pyramid_demo():
    # 图像金字塔
    image = cv.imread('/Users/dsj/Desktop/timg.jpeg')
    level = 3
    temp = image.copy()
    pyramid_images = []
    for i in range(level):
        dst = cv.pyrDown(temp)
        pyramid_images.append(dst)
        cv.imshow(f'pyramid_down_{i}', dst)
        temp = dst.copy()
    return pyramid_images


def laplace_demo():
    # 拉普拉斯金字塔
    image = cv.imread('/Users/dsj/Desktop/timg.jpeg')
    pyramid_images = pyramid_demo()
    level = len(pyramid_images)
    for i in range(level - 1, -1, -1):
        if (i - 1) < 0:
            expand = cv.pyrUp(pyramid_images[i], dstsize=image.shape[:2][::-1])
            lpls = cv.subtract(image, expand)
            cv.imshow(f'lpls_{i}', lpls)
        else:
            expand = cv.pyrUp(pyramid_images[i], dstsize=pyramid_images[i - 1].shape[:2][::-1])
            lpls = cv.subtract(pyramid_images[i - 1], expand)
            cv.imshow(f'lpls_{i}', lpls)


def sobel_demo():
    # 图像梯度
    image = cv.imread('/Users/dsj/Desktop/timg.jpeg')
    grad_x = cv.Sobel(image, cv.CV_32F, 1, 0)
    grad_y = cv.Sobel(image, cv.CV_32F, 0, 1)
    # cv.Scharr() 是Sobel算子的增强版
    gradx = cv.convertScaleAbs(grad_x)
    grady = cv.convertScaleAbs(grad_y)
    cv.imshow("gradient-x", gradx)
    cv.imshow("gradient-y", grady)
    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
    cv.imshow("gradient", gradxy)


def laplacian_demo():
    # 图像梯度，拉普拉斯算子
    image = cv.imread('/Users/dsj/Desktop/timg.jpeg')
    # dst = cv.Laplacian(image, cv.CV_32F)

    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])  # 自定义核
    dst = cv.filter2D(image, cv.CV_32F, kernel=kernel)
    lpls = cv.convertScaleAbs(dst)
    cv.imshow("laplacian_demo", lpls)


def canny_demo():
    # Canny边缘提取
    image = cv.imread('/Users/dsj/Desktop/timg.jpeg')
    blurred = cv.GaussianBlur(image, (3, 3), 0)  # 1. 高斯模糊
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)  # 2. 灰度
    grad_x = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
    grad_y = cv.Sobel(gray, cv.CV_16SC1, 0, 1)  # 3. 梯度
    edg_output0 = cv.Canny(grad_x, grad_y, 50, 150)  # 4. 边缘
    edg_output1 = cv.Canny(gray, 50, 150)  # 4. 效果一样
    # dst0 = cv.bitwise_and(image, image, mask=edg_output0)
    # dst1 = cv.bitwise_and(image, image, mask=edg_output1)
    cv.imshow('demo0', edg_output0)
    cv.imshow('demo1', edg_output1)


def line_detection():
    image = cv.imread('/Users/dsj/Desktop/timg3.jpeg')
    blurred = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    edg = cv.Canny(gray, 50, 150)
    lines = cv.HoughLines(edg, 1, np.pi / 180, 200)
    # cv.imshow('edg', edg)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a + rho
        y0 = b + rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv.imshow('line', image)


def line_detect_possible_demo():
    image = cv.imread('/Users/dsj/Desktop/timg3.jpeg')
    blurred = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    edg = cv.Canny(gray, 50, 150, apertureSize=3)
    cv.imshow('edg', edg)
    lines = cv.HoughLinesP(edg, 1, np.pi / 180, 200, minLineLength=50, maxLineGap=15)
    for line in lines:
        (x1, y1, x2, y2) = line[0]
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv.imshow('line', image)


def circles_detect():
    image = cv.imread('/Users/dsj/Desktop/circle.jpeg')
    dst = cv.pyrMeanShiftFiltering(image, 0, 100)
    gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 200, param1=50, param2=30, minRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)
        cv.circle(image, (i[0], i[1]), 2, (255, 0, 0), 2)
    cv.imshow('circles', image)


def main():
    # create_img()
    # video_demo()
    # pic_demo('/Users/dsj/Desktop/timg.jpeg')
    # extract_object_demo('/Users/dsj/Desktop/timg.jpeg')
    # calculate_pic()
    # logic_demo()
    # contrast_brightness_demo()
    # roi_demo()
    # fill_color_demo()
    # fill_binary()
    # blur_demo()
    # epf()
    # binary_image()
    # pyramid_demo()
    # laplace_demo()
    # sobel_demo()
    # laplacian_demo()
    # canny_demo()
    # line_detection()
    # line_detect_possible_demo()
    circles_detect()
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
