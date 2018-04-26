# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import cv2 as cv
import os


# Define a function to show image through 48*48 pixels
def show(img):
    show_image = img.reshape(48, 48)
    print(show_image)
    print(show_image.shape)
    cv.imshow('image', show_image)
    cv.waitKey(0)


def parse_images(data):
    pixels_values = data.pixels.str.split(" ").tolist()
    pixels_values = pd.DataFrame(pixels_values, dtype=int)
    images = pixels_values.values
    images = images.astype(np.uint8)
    return images


def save_data(dir_path, images, labels):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for i in range(len(images)):
        image = images[i].reshape(48, 48)
        label = emotion[labels[i]]
        image_path = os.path.join(dir_path, label)
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        image_path = os.path.join(image_path, str(i) + '.png')
        cv.imwrite(image_path, image)


def read_data(file_path):
    data = pd.read_csv(file_path)
    print(data.shape)
    print(data.head())
    print(np.unique(data["Usage"].values.ravel()))
    print('The number of training data set is %d' % (len(data[data.Usage == "Training"])))
    print('The number of validation data set is %d' % (len(data[data.Usage == "PrivateTest"])))
    train_data = data[data.Usage == "Training"]
    valid_data = data[data.Usage == "PrivateTest"]
    train_images = parse_images(train_data)
    valid_images = parse_images(valid_data)
    train_labels = train_data["emotion"].values.ravel()
    valid_labels = valid_data["emotion"].values.ravel()
    labels_count = np.unique(train_labels).shape[0]
    print(np.unique(train_labels))
    print('The number of different facial expressions is %d' % labels_count)

    # show one image
    # show(train_images[8])

    print('Start generating images...')
    save_data('fer2013/train', train_images, train_labels)
    save_data('fer2013/valid', valid_images, valid_labels)
    print('Completed.')


if __name__ == '__main__':
    image_pixels = 2304
    image_width = 48
    image_height = 48

    emotion = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    # emotion = {0:'愤怒', 1:'厌恶', 2:'恐惧', 3:'高兴', 4:'悲伤', 5:'惊讶', 6: '无表情'}

    read_data('fer2013/fer2013.csv')
