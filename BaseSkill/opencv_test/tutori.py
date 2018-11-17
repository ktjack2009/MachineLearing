import cv2 as cv


def get_image_info(img):
    print(img.size)
    print(img.shape)
    print(img.dtype)  # uint8 无符号8bit


def video_demo():
    # 读取视频
    capture = cv.VideoCapture(0)
    cv.namedWindow('video', cv.WINDOW_AUTOSIZE)
    while True:
        ret, frame = capture.read()
        frame = cv.flip(frame, 1)  # 摄像头默认左右颠倒，code=1镜像，code=0左右上下颠倒，code=-1上下颠倒
        # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)   # 转换未灰度图
        cv.imshow("video", frame)
        c = cv.waitKey(50)  # 延时50ms后返回按键，如果未按，返回-1
        print(c)
        if c == 27:  # esc ascii码的值
            break
        # elif c == 97:
        #     cv.imwrite('/home/1.png', frame)   # 保存图片
    cv.destroyAllWindows()


def pic_demo(path):
    src = cv.imread(path)  # 返回一个ndarray对象
    cv.namedWindow('input image', cv.WINDOW_AUTOSIZE)  # 设置图像显示窗口
    cv.imshow('input image', src)
    cv.waitKey(0)
    cv.destroyAllWindows()


video_demo()
