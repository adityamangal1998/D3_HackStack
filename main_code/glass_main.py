# https://www.kaggle.com/lebegus/glasses-detection/notebook#5)-Prediction
import numpy as np
import tensorflow as tf
import keras
import cv2
import glass_utils as utils

model = keras.models.load_model('glass_model')
class_names = ['glass', 'no glass']


def main(image):
    image = cv2.resize(image, (160, 160))
    test = [image]
    X_test = np.asarray(test)
    Y_test = model.predict(X_test)
    predictions = tf.nn.sigmoid(Y_test[0])
    predictions = tf.where(predictions < 0.5, 0, 1)
    text = class_names[predictions.numpy()[0]]
    return text
