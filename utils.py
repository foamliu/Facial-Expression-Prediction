import numpy as np
import cv2
from resnet_101 import resnet101_model


def load_emotion_model(model_path):
    img_width, img_height = 224, 224
    num_channels = 3
    num_classes = 7

    emotion_model = resnet101_model(img_height, img_width, num_channels, num_classes)
    emotion_model.load_weights(model_path, by_name=True)
    return emotion_model


def draw_bounding_box(image, coordinates, color):
    x, y, w, h = coordinates
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)


def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return x - x_off, x + width + x_off, y - y_off, y + height + y_off


def draw_text(image, coordinates, text, color, x_offset=0, y_offset=0,
              font_scale=2, thickness=2):
    x, y = coordinates[:2]
    cv2.putText(image, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)


def get_color(emotion, prob):
    if emotion.lower() == 'angry':
        color = prob * np.asarray((0, 0, 255))
    elif emotion.lower() == 'sad':
        color = prob * np.asarray((255, 0, 0))
    elif emotion.lower() == 'happy':
        color = prob * np.asarray((0, 255, 255))
    elif emotion.lower() == 'surprise':
        color = prob * np.asarray((255, 255, 0))
    elif emotion.lower() == 'fear':
        color = prob * np.asarray((255, 255, 255))
    elif emotion.lower() == 'disgust':
        color = prob * np.asarray((255, 0, 255))
    else:
        color = prob * np.asarray((0, 255, 0))
    return color


def draw_str(dst, target, s):
    x, y = target
    cv2.putText(dst, s, (x + 1, y + 1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
