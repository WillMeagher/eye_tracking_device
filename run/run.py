import cv2
import numpy as np
import math
from usb import mouse
from camera import camera
from ml import gaze_estimation, eye_status
from tools import speed_test, input_check
from config import *

SCREEN_WIDTH_INCHES = 23.375
SCREEN_HEIGHT_INCHES = 13.25
DISTANCE_TO_SCREEN_INCHES = 30

def main():
    cam = camera.Camera()

    gaze_model = gaze_estimation.GazeEstimator(config["models_path"] + config["gaze_estimation_models"])
    eye_model = eye_status.EyeStatusEstimator(config["models_path"] + config["eye_status_models"])

    speed_tester = speed_test.SpeedTest()

    while(True):
        frame = cam.get_frame()
        frame = prepare_frame(frame)

        eye_status_result = eye_model.run(frame)

        if eye_status_result[0] == 1 and eye_status_result[1] == 1:
            gaze_estimation_result = gaze_model.run(frame)

            x, y = get_pos(gaze_estimation_result)
            # mouse.move(x, y)
            # print("x=%6.2f" %gaze_estimation_result[0], "y=%6.2f" %gaze_estimation_result[1])

        else:
            # print("eye/s closed")
            pass

        speed_tester.loop()

        if input_check.check("q"):
            break

    cam.release()


def cleanup():
    input_check.exit()


def prepare_frame(img):

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img = img[41:56, 16:-16]
    left = img[:, :30]
    right = img[:, -30:]
    eyes = np.concatenate((left, right), axis=1)

    eyes = eyes / 255.0

    return eyes


def get_pos(angles):
    x_mid = DISTANCE_TO_SCREEN_INCHES * math.tan(angles[0])
    y_mid = DISTANCE_TO_SCREEN_INCHES * math.tan(angles[1])

    x = SCREEN_WIDTH_INCHES / 2 + x_mid
    y = SCREEN_HEIGHT_INCHES / 2 + y_mid

    return x, y


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup()