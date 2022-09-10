import numpy as np
from ml import model

class EyeStatusEstimator(model.Model):

    def run(self, img):
        left, right = get_eyes(img)

        prediction_l = super().run(left)
        prediction_r = super().run(right)

        return prediction_l, prediction_r


    def process_prediction(self, prediction):
        prediction = prediction[0][0]

        prediction = round(prediction)

        return prediction


    def prepare_img(self, img):
        return np.reshape(np.array([img], dtype=np.float32), (1, 15, 30, 1))


def get_eyes(img):
    left = img[:, :img.shape[1] // 2]
    right = img[:, img.shape[1] // 2:]

    return left, right