from resnet_152 import resnet152_model
from keras.preprocessing.image import ImageDataGenerator

IMG_WIDTH, IMG_HEIGHT = 224, 224
TRAIN_DATA = 'fer2013/train'
VALID_DATA = 'fer2013/valid'
NUM_CLASSES = 7
NB_TRAIN_SAMPLES = 28709
NB_VALID_SAMPLES = 3589
BATCH_SIZE = 32

# build a classifier model
model = resnet152_model(IMG_HEIGHT, IMG_WIDTH, 3, NUM_CLASSES)

# prepare data augmentation configuration
train_data_gen = ImageDataGenerator(featurewise_std_normalization=True, zoom_range=0.2, rotation_range=20,
                                    width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
valid_data_gen = ImageDataGenerator(featurewise_std_normalization=True, zoom_range=0.2, rotation_range=20,
                                    width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)

train_generator = train_data_gen.flow_from_directory(TRAIN_DATA, (IMG_WIDTH, IMG_HEIGHT), batch_size=BATCH_SIZE,
                                                     class_mode='categorical')
valid_generator = valid_data_gen.flow_from_directory(VALID_DATA, (IMG_WIDTH, IMG_HEIGHT), batch_size=BATCH_SIZE,
                                                     class_mode='categorical')

# fine tune the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=NB_TRAIN_SAMPLES // BATCH_SIZE,
    validation_data=valid_generator,
    validation_steps=NB_VALID_SAMPLES // BATCH_SIZE,
    epochs=10)

model.save_weights("model.h5")
