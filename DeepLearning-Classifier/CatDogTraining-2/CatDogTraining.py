from keras import optimizers
from keras import losses
from keras.models import Sequential, load_model, Model
from keras.layers import *
import h5py
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import os
import time
from keras.callbacks import CSVLogger, TensorBoard, LambdaCallback

try:
    train_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        directory='dataset/train',
        target_size=(224, 224),
        batch_size=1,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )

    test_generator = test_datagen.flow_from_directory(
        directory='dataset/test',
        target_size=(224, 224),
        batch_size=1,
        class_mode=None,
        shuffle=False
    )
except Exception as e:
    print(str(e))

NUMBER_OF_CLASSES = len(train_generator.class_indices)

STEP_SIZE_TRAIN = train_generator.n

model = Sequential()

model.add(Conv2D(1, kernel_size=3, padding='same', input_shape=(224, 224, 3)))
model.add(GaussianNoise(0.25))

model.add(Conv2D(8, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
model.add(GaussianNoise(0.25))
model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(256, kernel_size=3, padding='same', activation='relu'))
model.add(GaussianNoise(0.25))
model.add(Conv2D(512, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.1))
model.add(GaussianNoise(0.25))
model.add(Dense(512, activation='relu'))
model.add(Dense(NUMBER_OF_CLASSES, activation='sigmoid'))
model.summary()

opt = optimizers.SGD(learning_rate=1e-4)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

def epoch_end(epoch, logs):
    filenames = test_generator.filenames
    test_generator.reset()
    pred = model.predict_generator(generator=test_generator,
                                   steps=test_generator.n // test_generator.batch_size,
                                   verbose=1)

    predicted_class_indices = np.argmax(pred, axis=1)
    acc = np.sum(predicted_class_indices == test_generator.classes) * 100 / len(filenames)
    print('test_accuracy:', '{0:.2f}'.format(acc))

    if acc >= 70:
        new_path = 'epoch_' + str(epoch) + '_accuracy_' + '{0:.2f}'.format(acc)
        model.save('trainedmodels/vgg16_' + new_path + '.h5')
        model.save_weights('trainedmodels/vgg16_' + new_path + '_weights_.h5')

testmodelcb = LambdaCallback(on_epoch_end=epoch_end)
model.fit(x=train_generator,
          steps_per_epoch=STEP_SIZE_TRAIN,
          epochs=300,
          callbacks=[testmodelcb])