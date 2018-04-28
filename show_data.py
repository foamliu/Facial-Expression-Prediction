# -*- coding: utf-8 -*-

import pandas as pd
import cv2 as cv
import numpy as np
import random


def parse_images(data):
    pixels_values = data.pixels.str.split(" ").tolist()
    pixels_values = pd.DataFrame(pixels_values, dtype=int)
    images = pixels_values.values
    images = images.astype(np.uint8)
    return images


def read_data(file_path, usage):
    data = pd.read_csv(file_path)
    print(data.shape)
    print(data.head())
    data = data[data.Usage == usage]
    images = parse_images(data)
    return images


def show_data(images, num_rows, num_cols):
    print(images.shape)
    num_images = len(images)
    print('The number of data set is %d' % num_images)
    canvas = np.zeros((num_rows * unit_height, num_cols * unit_width), dtype=np.uint8)
    print(canvas.shape)
    index_dict = random.sample(range(num_images), (num_rows * num_cols))
    for i in range(num_rows):
        for j in range(num_cols):
            index = index_dict[i * num_cols + j]
            image = images[index]
            image = image.reshape(unit_width, unit_height)
            i1 = i * unit_height
            i2 = (i + 1) * unit_height
            j1 = j * unit_width
            j2 = (j + 1) * unit_width
            #print(i1, i2, j1, j2)
            canvas[i1:i2, j1:j2] = image
    cv.imwrite('images/random.png', canvas)
    cv.imshow("data", canvas)
    cv.waitKey(0)


if __name__ == '__main__':
    unit_width = unit_height = 48
    show_data(read_data('fer2013/fer2013.csv', "Training"), 10, 20)
