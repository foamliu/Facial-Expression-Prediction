# Facial-Expression-Prediction


This repository is to do facial expression prediction by fine-tuning ResNet-50 against FER-2013 Faces Database.


## Dependencies

- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
- [Keras](https://keras.io/#installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)

## Dataset

We use the FER-2013 Faces Database, a set of 28,709 pictures of people displaying 7 emotional expressions (angry, disgusted, fearful, happy, sad, surprised and neutral).

You can get it on [Kraggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).

Please extract the package so that fer2013.csv is in fer2013 folder.

## ImageNet Pretrained Models

Please download [ResNet-50](https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5) into imagenet_models folder.

## Usage

### Data Pre-processing
```bash
$ python pre-process.py
```

### Train
```bash
$ python train.py
```

### Predict
```bash
$ python predict.py --i [image_path]
```