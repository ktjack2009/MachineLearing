import os
import cv2 as cv
import numpy as np

BaseDir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(BaseDir, 'haarcascade_frontalface_alt_tree.xml')


def face_detect_demo(image):
    # 人脸识别
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    face_detector = cv.CascadeClassifier(path)
    faces = face_detector.detectMultiScale(gray, 1.01, 1)
    for (x, y, w, h) in faces:
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv.imshow('camera', image)


def camera_capture():
    capture = cv.VideoCapture(0)
    while True:
        ret, image = capture.read()
        image = cv.flip(image, 1)
        face_detect_demo(image)
        c = cv.waitKey(50)
        if c == 27:
            break
    cv.destroyAllWindows()


# camera_capture()
image = cv.imread('/Users/dsj/Desktop/timg1.jpeg')
face_detect_demo(image)
cv.waitKey(0)
cv.destroyAllWindows()
