from keras import optimizers
from keras import losses
from keras.models import Sequential, load_model, Model
from keras.layers import *
import h5py
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import time
from keras.callbacks import LambdaCallback


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
        class_mode='categorical',
        shuffle=False
    )
except Exception as e:
    print(str(e))

NUMBER_OF_CLASSES = len(train_generator.class_indices)

STEP_SIZE_TRAIN = train_generator.samples
STEP_SIZE_TEST = test_generator.samples


model = Sequential()
model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu', input_shape=(224, 224, 3)))
model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(128, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(NUMBER_OF_CLASSES, activation='softmax'))
model.summary()

opt = optimizers.Adam(learning_rate=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

def epoch_end(epoch, logs):
    filenames = test_generator.filenames
    test_generator.reset()
    pred = model.predict(x=test_generator,
                         steps=STEP_SIZE_TEST,
                         verbose=1)

    predicted_class_indices = np.argmax(pred, axis=1)
    acc = np.sum(predicted_class_indices == test_generator.classes) * 100 / len(filenames)
    print('test_accuracy:', '{0:.2f}'.format(acc))

    if acc >= 90:
        new_path = 'epoch_' + str(epoch) + '_accuracy_' + '{0:.2f}'.format(acc)
        model.save('trainedmodels/vgg16_' + new_path + '.h5')
        model.save_weights('trainedmodels/vgg16_' + new_path + '_weights_.h5')

testmodelcb = LambdaCallback(on_epoch_end=epoch_end)

model.fit(train_generator,
          steps_per_epoch=STEP_SIZE_TRAIN,
          epochs=300,
          callbacks=[testmodelcb])