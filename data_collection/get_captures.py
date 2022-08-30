import cv2
import time
import os

CAPTURES_FILE_PATH = "/home/pi/Downloads/eye_tracking_device/data_collection/captures/"

# make captures folder if not exists
if not os.path.exists(CAPTURES_FILE_PATH):
    os.makedirs(CAPTURES_FILE_PATH)

cur_frames = 0

# start video
cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 360)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

while True:
    ret, frame = cap.read()

    cv2.imwrite(CAPTURES_FILE_PATH + str(int(time.time() * 1000)) + ".jpg", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
