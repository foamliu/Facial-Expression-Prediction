# Facial-Expression-Prediction


This repository is using convolutional neural network to do facial expression prediction.


## Dependencies

- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
- [Keras](https://keras.io/#installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)

## Dataset

We use the FER-2013 Faces Database, a set of 28,709 pictures of people displaying 7 emotional expressions (angry, disgusted, fearful, happy, sad, surprised and neutral).

You can get it on [Kraggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).

Please extract the package so that the folder structure looks like:

FACIAL-EXPRESSION-PREDICTION
│   predict.py
│   pre-process.py
│   README.md
│   train.py
├───fer2013
│       fer2013.bib
│       fer2013.csv
│       README

## Usage

### Train
```bash
$ python train.py
```

### Predict
```bash
$ python predict.py --i [image_path]
```