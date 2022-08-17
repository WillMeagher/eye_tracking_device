import cv2
import time
import numpy as np
from tflite_runtime.interpreter import Interpreter

def main():
    num_frames = 100
    cur_frames = 0

    global output_details
    global input_details
    global interpreter

    MODEL_PATH = '/home/pi/Downloads/eye_tracking_device/ml/models/'
    MODEL_NAME = 'model_3.38.tflite'
    
    interpreter = Interpreter(model_path=MODEL_PATH + MODEL_NAME)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # start video
    cap = cv2.VideoCapture(0)

    cap.set(3, 640)
    cap.set(4, 360)

    start = time.time()

    while True:
        ret, img = cap.read()

        eyes = prepare_img(img)
        
        prediction = predict(eyes)

        prediction = process_prediction(prediction)
        
        print("x=%6.2f" %prediction[0], "y=%6.2f" %prediction[1])

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

    img = img[205:280, 80:-80]
    left = img[:, :150]
    right = img[:, -150:]
    eyes = np.concatenate((left, right), axis=1)

    eyes = cv2.resize(eyes, (0,0), fx=0.2, fy=0.2)

    eyes = eyes / 255.0

    return eyes


def predict(img):
    input_data = np.reshape(np.array([img], dtype=np.float32), (1, 15, 60, 1))

    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    return output_data


def process_prediction(prediction):
    prediction = prediction[0]

    prediction = (prediction * 90) - 45

    return prediction


if __name__ == '__main__':
    main()
