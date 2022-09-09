import numpy as np
from tflite_runtime.interpreter import Interpreter

class GazeEstimator:

    def __init__(self, model_path):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()


    def run(self, img):
        prediction = self.predict(img)
        prediction = self.process_prediction(prediction)

        return prediction


    def predict(self, img):
        input_data = np.reshape(np.array([img], dtype=np.float32), (1, 15, 60, 1))

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        return output_data


    def process_prediction(self, prediction):
        prediction = prediction[0]

        prediction = (prediction * 90) - 45

        return prediction