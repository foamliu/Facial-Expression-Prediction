# 面部表情识别


基于 FER-2013 人脸表情数据集对 ResNet-101 进行微调来进行面部表情识别。


## 依赖

- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
- [Keras](https://keras.io/#installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)

## 数据集

我使用了 FER-2013 人脸表情数据集，这是一组显示7种情绪表达（愤怒，厌恶，恐惧，快乐，伤心，惊讶和中立）的35,887张照片。

 ![image](https://github.com/foamliu/Facial-Expression-Prediction/raw/master/images/random.png)

你可以从 [Kraggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)下载数据集，别忘了把fer2013.csv放在fer2013文件夹中。

## ImageNet 预训练模型

下载 [ResNet-101](https://drive.google.com/file/d/0Byy2AcGyEVxfTmRRVmpGWDczaXM/view?usp=sharing) 放在 models 文件夹中。

我在微调ResNet-152时遇到了OOM错误，你不妨试一下。

## 如何使用

### 数据预处理
解压 28,709 张训练图片, 和 3,589 张验证图片:
```bash
$ python pre-process.py
```
  
### 训练
```bash
$ python train.py
```

If you want to visualize during training, run in your terminal:
```bash
$ tensorboard --logdir path_to_current_dir/logs
```

 ![image](https://github.com/foamliu/Facial-Expression-Prediction/raw/master/images/train.png)



### 结果分析
将最好的模型重命名为“Model.best.hdf5”，将其放在“models”文件夹中，并使用3,589个测试集图片进行结果分析：
```bash
$ python analyze.py
```

#### 测试集准确率: 
**71.22%**

#### 混淆矩阵:

 ![image](https://github.com/foamliu/Facial-Expression-Prediction/raw/master/images/confusion_matrix_not_normalized.png)

 ![image](https://github.com/foamliu/Facial-Expression-Prediction/raw/master/images/confusion_matrix_normalized.png)


### Demo
下载 [预训练模型](https://github.com/foamliu/Facial-Expression-Prediction/releases/download/v1.0/model.best.hdf5) 放在 "models" 目录下然后执行:

```bash
$ python demo.py --v [video_path]
```

 ![image](https://github.com/foamliu/Facial-Expression-Prediction/raw/master/images/demo.gif)