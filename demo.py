# import the necessary packages
import cv2 as cv
import numpy as np
import argparse
import dlib
import time
import keras.backend as K
from utils import load_emotion_model
from utils import apply_offsets
from utils import draw_bounding_box
from utils import draw_text
from utils import get_color
from utils import draw_str
from console_progressbar import ProgressBar


def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return x, y, w, h


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the video file")
    args = vars(ap.parse_args())

    video = args["video"]
    if video is None:
        video = 'video/test.mp4'

    img_width, img_height = 224, 224
    num_channels = 3
    num_classes = 7

    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    # class_names = ['愤怒', '厌恶', '恐惧', '高兴', '悲伤', '惊讶', '无表情']

    detector = dlib.get_frontal_face_detector()
    emotion_model = load_emotion_model('models/model.best.hdf5')

    # initialize the camera and grab a reference to the raw camera capture
    cap = cv.VideoCapture(video)

    ret, frame = cap.read()
    height, width = frame.shape[:2]

    fourcc = cv.VideoWriter_fourcc(*'MPEG')
    out = cv.VideoWriter('video/output.avi', fourcc, 24.0, (width, width))

    start = time.time()
    frame_idx = 0
    num_frames = 984
    pb = ProgressBar(total=100, prefix='Processing video', suffix='', decimals=3, length=50, fill='=')
    # capture frames from the camera
    while ret:
        frame = frame[100:100 + width, :]
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.cvtColor(gray, cv.COLOR_GRAY2RGB)
        faces = detector(gray, 1)

        for rect in faces:
            (x, y, w, h) = rect_to_bb(rect)
            x1, x2, y1, y2 = apply_offsets((x, y, w, h), (10, 10))
            gray_face = gray[y1:y2, x1:x2]
            gray_face = cv.resize(gray_face, (img_height, img_width))
            gray_face = np.expand_dims(gray_face, 0)
            preds = emotion_model.predict(gray_face)
            prob = np.max(preds)
            class_id = np.argmax(preds)
            emotion = class_names[class_id]
            #
            # print(emotion)

            color = get_color(emotion, prob)
            draw_bounding_box(image=frame, coordinates=(x1, y1, x2 - x1, y2 - y1), color=color)
            draw_text(image=frame, coordinates=(x1, y1, x2 - x1, y2 - y1), color=color, text=emotion)
            out.write(frame)

        end = time.time()
        seconds = end - start
        fps = 1.0 / seconds
        draw_str(frame, (20, 20), 'frame_idx: %d, fps: %.2f' % (frame_idx, fps))
        print('fps: %.2f' % fps)
        pb.print_progress_bar(frame_idx * 100 / num_frames)

        # show the frame
        # cv.imshow("Frame", frame)

        frame_idx += 1
        start = time.time()

        # if the `q` key was pressed, break from the loop
        key = cv.waitKey(1)
        if key == ord("q"):
            break

        ret, frame = cap.read()

    cap.release()
    out.release()
    # cv.destroyAllWindows()
    K.clear_session()
