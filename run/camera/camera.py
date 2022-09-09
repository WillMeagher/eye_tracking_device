import cv2

class Camera:
    def __init__(self, cap_id=0):
        self.cap_id = cap_id
        self.cap = cv2.VideoCapture(self.cap_id)

        self.cap.set(3, 128)
        self.cap.set(4, 72)

    def get_frame(self):
        ret, frame = self.cap.read()
        return frame

    def release(self):
        self.cap.release()

    def __del__(self):
        self.release()