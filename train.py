# -*- coding: utf-8 -*-

import keras

from resnet_101 import resnet101_model
from keras.preprocessing.image import ImageDataGenerator

IMG_WIDTH, IMG_HEIGHT = 224, 224
TRAIN_DATA = 'fer2013/train'
VALID_DATA = 'fer2013/valid'
NUM_CLASSES = 7
NB_TRAIN_SAMPLES = 28709
NB_VALID_SAMPLES = 3589
BATCH_SIZE = 32


if __name__ == '__main__':
    # build a classifier model
    model = resnet101_model(IMG_HEIGHT, IMG_WIDTH, 3, NUM_CLASSES)

    # prepare data augmentation configuration
    train_data_gen = ImageDataGenerator(rescale=1. / 255, zoom_range=0.2, rotation_range=20,
                                        width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
    valid_data_gen = ImageDataGenerator(rescale=1. / 255, zoom_range=0.2, rotation_range=20,
                                        width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)

    train_generator = train_data_gen.flow_from_directory(TRAIN_DATA, (IMG_WIDTH, IMG_HEIGHT), batch_size=BATCH_SIZE,
                                                         class_mode='categorical')
    valid_generator = valid_data_gen.flow_from_directory(VALID_DATA, (IMG_WIDTH, IMG_HEIGHT), batch_size=BATCH_SIZE,
                                                         class_mode='categorical')

    tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    # fine tune the model
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=NB_TRAIN_SAMPLES // BATCH_SIZE,
        validation_data=valid_generator,
        validation_steps=NB_VALID_SAMPLES // BATCH_SIZE,
        epochs=10,
        callbacks=[tbCallBack])

    model.save_weights("model.h5")
