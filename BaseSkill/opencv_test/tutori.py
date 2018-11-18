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
    epf()
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
