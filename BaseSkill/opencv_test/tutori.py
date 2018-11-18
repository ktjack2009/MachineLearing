import cv2 as cv
import numpy as np


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

    cv.waitKey(0)
    cv.destroyAllWindows()


def create_img():
    src = np.zeros((400, 400, 3), dtype=np.uint8)
    src[:, :, 0] = np.ones((400, 400)) * 255
    src[:, :, 2] = np.ones((400, 400)) * 255
    cv.namedWindow('input image', cv.WINDOW_AUTOSIZE)  # 设置图像显示窗口
    cv.imshow('input image', src)
    cv.waitKey(0)
    cv.destroyAllWindows()


# video_demo()
pic_demo('/Users/dsj/Desktop/timg.jpeg')
# create_img()
