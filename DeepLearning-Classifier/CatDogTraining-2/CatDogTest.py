import os
import sys
from tensorflow.keras.models import Sequential, load_model
import cv2
from datetime import datetime
import time
import numpy as np

try:
    class_indices = ['cat', 'dog']
    model = Sequential()
    model = load_model('trainedmodels/vgg16_epoch_13_accuracy_84.55.h5')
    model.load_weights('trainedmodels/vgg16_epoch_13_accuracy_84.55_weights_.h5')
    folder = 'dataset/testImage'
    images_count = len(os.listdir(folder))
    print('file count : ', images_count)

    for file in os.listdir(folder):
        path = folder + '/' + file
        img = cv2.imread(path, 1)
        img = cv2.resize(img, (224, 224))
        img = np.array([img]).reshape((1, 224, 224, 3))
        res = model.predict(img)
        pred_name = class_indices[np.argmax(res)]
        print('result: ', pred_name, 'file: ', file)

except Exception as e:
    print(str(e))