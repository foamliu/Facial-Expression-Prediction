# Facial Expression Prediction


This repository is to do facial expression prediction by fine-tuning ResNet-101 with FER-2013 Faces Database.


## Dependencies

- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
- [Keras](https://keras.io/#installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)

## Dataset

I use the FER-2013 Faces Database, a set of 35,887 pictures of people displaying 7 emotional expressions (angry, disgusted, fearful, happy, sad, surprised and neutral).

 ![image](https://github.com/foamliu/Facial-Expression-Prediction/raw/master/images/random.png)

You can get it from [Kraggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data), make sure fer2013.csv is in fer2013 folder.

## ImageNet Pretrained Models

Download [ResNet-101](https://drive.google.com/file/d/0Byy2AcGyEVxfTmRRVmpGWDczaXM/view?usp=sharing) into imagenet_models folder.

I met OOM error when fine-tuning ResNet-152, you may want to have a try.

## Usage

### Data Pre-processing
Extract 28,709 images [Usage='Training'] for training, and 3,589 [Usage='PublicTest'] for validation:
```bash
$ python pre-process.py
```
 When complete, folder structure looks like:
 
 ![image](https://github.com/foamliu/Facial-Expression-Prediction/raw/master/images/data.PNG)
 
### Train
```bash
$ python train.py
```
 ![image](https://github.com/foamliu/Facial-Expression-Prediction/raw/master/images/train.png)

If you want to visualize during training, run in your terminal:
```bash
$ tensorboard --logdir path_to_current_dir/logs
```

### Analysis
Use 3,589 images [Usage='PrivateTest'] for result analysis.


### Predict
```bash
$ python predict.py --i [image_path]
```