import cv2
from .config import CAMERA_SOURCE


class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_SOURCE)

        if not self.cap.isOpened():
            raise RuntimeError("Camera not accessible")

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.cap.release()