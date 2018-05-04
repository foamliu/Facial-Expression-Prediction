# import the necessary packages
import cv2 as cv
import numpy as np
import argparse
import dlib
from utils import load_emotion_model
from utils import apply_offsets
from utils import draw_bounding_box
from utils import draw_text
from utils import get_color
from utils import draw_str


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
    # emotion = {0:'愤怒', 1:'厌恶', 2:'恐惧', 3:'高兴', 4:'悲伤', 5:'惊讶', 6: '无表情'}

    detector = dlib.get_frontal_face_detector()
    emotion_model = load_emotion_model('models/model.best.hdf5')

    # initialize the camera and grab a reference to the raw camera capture
    cap = cv.VideoCapture(video)

    ret, frame = cap.read()
    height, width = frame.shape[:2]

    fourcc = cv.VideoWriter_fourcc(*'MPEG')
    out = cv.VideoWriter('video/output.avi', fourcc, 24.0, (width, height))

    frame_idx = 0
    # capture frames from the camera
    while ret:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.cvtColor(gray, cv.COLOR_GRAY2RGB)
        faces = detector(gray, 1)

        for rect in faces:
            (x, y, w, h) = rect_to_bb(rect)
            x1, x2, y1, y2 = apply_offsets((x, y, w, h), (20, 40))
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

        draw_str(frame, (20, 20), 'frame_idx: %d.' % frame_idx)

        # show the frame
        cv.imshow("Frame", frame)

        frame_idx += 1

        # if the `q` key was pressed, break from the loop
        key = cv.waitKey(1)
        if key == ord("q"):
            break

        ret, frame = cap.read()

    cap.release()
    out.release()
    cv.destroyAllWindows()
