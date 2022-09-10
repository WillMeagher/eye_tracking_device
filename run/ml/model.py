import numpy as np
import os
from tflite_runtime.interpreter import Interpreter

class Model:

    def __init__(self, models_path):
        self.interpreters = []
        self.input_details = []
        self.output_details = []

        for model_name in os.listdir(models_path):
            interpreter = Interpreter(model_path=models_path + model_name)
            self.interpreters.append(interpreter)
            interpreter.allocate_tensors()

            self.input_details.append(interpreter.get_input_details())
            self.output_details.append(interpreter.get_output_details())


    def run(self, img):
        input_data = self.prepare_img(img)
        prediction = self.predict(input_data)
        prediction = self.process_prediction(prediction)

        return prediction


    def predict(self, input_data):
        output_data = []

        for i in range(len(self.interpreters)):
            self.interpreters[i].set_tensor(self.input_details[i][0]['index'], input_data)
            self.interpreters[i].invoke()

            output_data.append(self.interpreters[i].get_tensor(self.output_details[i][0]['index']))

        output_data = np.array(output_data)

        output_data = np.mean(output_data, axis=0)

        return output_data


    def prepare_img(self, img):
        pass

    def process_prediction(self, prediction):
        pass

def reduce(val):
    if isinstance(val, (list, tuple, np.ndarray)) and len(val) == 1:
        return reduce(val[0])
    else:
        return val