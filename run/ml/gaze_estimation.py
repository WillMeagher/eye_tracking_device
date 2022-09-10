import numpy as np
from ml import model

class GazeEstimator(model.Model):

    def prepare_img(self, img):
        return np.reshape(np.array([img], dtype=np.float32), (1, 15, 60, 1))


    def process_prediction(self, prediction):
        prediction = prediction[0]

        prediction = (prediction * 90) - 45

        return prediction