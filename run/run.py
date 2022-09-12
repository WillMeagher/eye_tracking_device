import cv2
import numpy as np
import math
from usb import mouse
from camera import camera
from ml import gaze_estimation, eye_status
from tools import speed_test, input_check, utils
from config import *

SCREEN_WIDTH_INCHES = 23.375
SCREEN_HEIGHT_INCHES = 13.25
INVERSE_SCREEN_WIDTH_INCHES = 0.04278
INVERSE_SCREEN_HEIGHT_INCHES = 0.07547
DISTANCE_TO_SCREEN_INCHES = 30
PI_OVER_180 = 0.01745

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
            # print("x=%6.2f" %gaze_estimation_result[0], "y=%6.2f" %gaze_estimation_result[1])

            x, y = get_pos(gaze_estimation_result)
            print("x=%6.2f" %x, "y=%6.2f" %y)
            # mouse.move(x, y)
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
    np.array(angles)
    angles = angles * PI_OVER_180

    from_middle_x = DISTANCE_TO_SCREEN_INCHES * utils.fast_tan(angles[0])
    from_middle_y = DISTANCE_TO_SCREEN_INCHES * utils.fast_tan(angles[1])

    x = from_middle_x * INVERSE_SCREEN_WIDTH_INCHES + .5
    y = from_middle_y * INVERSE_SCREEN_HEIGHT_INCHES + .5

    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, 1)

    return x, y


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup()