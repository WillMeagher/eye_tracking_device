from selectors import EpollSelector
import cv2
import time
import numpy as np
from tflite_runtime.interpreter import Interpreter

def main():
    num_frames = 100
    cur_frames = 0

    global output_details_eye
    global input_details_eye
    global interpreter_eye

    global output_details_gaze
    global input_details_gaze
    global interpreter_gaze

    MODEL_PATH = '/home/pi/Downloads/eye_tracking_device/ml/models/'
    MODEL_NAME_GAZE = 'model_3.38.tflite'
    MODEL_NAME_EYE = '1662069423699_0.00137_model.tflite'
    
    interpreter_gaze = Interpreter(model_path=MODEL_PATH + MODEL_NAME_GAZE)
    interpreter_gaze.allocate_tensors()

    input_details_gaze = interpreter_gaze.get_input_details()
    output_details_gaze = interpreter_gaze.get_output_details()


    interpreter_eye = Interpreter(model_path=MODEL_PATH + MODEL_NAME_EYE)
    interpreter_eye.allocate_tensors()

    input_details_eye = interpreter_eye.get_input_details()
    output_details_eye = interpreter_eye.get_output_details()

    # start video
    cap = cv2.VideoCapture(0)

    cap.set(3, 128)
    cap.set(4, 72)

    start = time.time()

    while True:
        ret, img = cap.read()

        eyes = prepare_img(img)

        left, right = get_eyes(eyes)

        left_prediction, right_prediction = predict_eye(left), predict_eye(right)
        left_prediction, right_prediction = process_prediction_eye(left_prediction), process_prediction_eye(right_prediction)

        if left_prediction == 1 and right_prediction == 1:

            prediction_gaze = predict_gaze(eyes)
            prediction_gaze = process_prediction_gaze(prediction_gaze)
            
            print("x=%6.2f" %prediction_gaze[0], "y=%6.2f" %prediction_gaze[1])

        else:
            print("No eyes detected")


        cur_frames += 1

        if cur_frames == num_frames:
            cur_frames = 0
            end = time.time()
            elapsed = end - start
            # print(str(num_frames / elapsed) + "fps")
            start = end

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def prepare_img(img):

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img = img[41:56, 16:-16]
    left = img[:, :30]
    right = img[:, -30:]
    eyes = np.concatenate((left, right), axis=1)

    # eyes = cv2.resize(eyes, (0,0), fx=0.2, fy=0.2)

    eyes = eyes / 255.0

    return eyes


def get_eyes(img):
    left = img[:, :img.shape[1] // 2]
    right = img[:, img.shape[1] // 2:]

    return left, right


def process_prediction_eye(prediction):
    prediction = prediction[0][0]

    prediction = round(prediction)

    return prediction


def predict_eye(eye):
    input_data = np.reshape(np.array([eye], dtype=np.float32), (1, 15, 30, 1))

    interpreter_eye.set_tensor(input_details_eye[0]['index'], input_data)
    
    interpreter_eye.invoke()

    output_data = interpreter_eye.get_tensor(output_details_eye[0]['index'])

    return output_data


def predict_gaze(img):
    input_data = np.reshape(np.array([img], dtype=np.float32), (1, 15, 60, 1))

    interpreter_gaze.set_tensor(input_details_gaze[0]['index'], input_data)
    
    interpreter_gaze.invoke()

    output_data = interpreter_gaze.get_tensor(output_details_gaze[0]['index'])

    return output_data


def process_prediction_gaze(prediction):
    prediction = prediction[0]

    prediction = (prediction * 90) - 45

    return prediction


if __name__ == '__main__':
    main()
