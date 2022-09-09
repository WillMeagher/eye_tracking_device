import numpy as np
from tflite_runtime.interpreter import Interpreter

class EyeStatusEstimator:

    def __init__(self, model_path):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()


    def run(self, img):
        left, right = get_eyes(img)

        prediction_l = self.predict(left)
        prediction_l = self.process_prediction(prediction_l)

        prediction_r = self.predict(right)
        prediction_r = self.process_prediction(prediction_r)

        return prediction_l, prediction_r


    def predict(self, img):
        input_data = np.reshape(np.array([img], dtype=np.float32), (1, 15, 30, 1))

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        return output_data

    
    def process_prediction(self, prediction):
        prediction = prediction[0][0]

        prediction = round(prediction)

        return prediction


def get_eyes(img):
    left = img[:, :img.shape[1] // 2]
    right = img[:, img.shape[1] // 2:]

    return left, right