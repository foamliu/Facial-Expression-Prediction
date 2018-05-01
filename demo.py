# import the necessary packages
import cv2 as cv
import numpy as np
import argparse
from utils import load_detection_model
from utils import load_emotion_model
from utils import detect_faces
from utils import apply_offsets
from utils import draw_bounding_box
from utils import draw_text
from utils import get_color

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the video file")
    args = vars(ap.parse_args())

    video = args["video"]
    if video is None:
        video = 'test.mp4'

    img_width, img_height = 224, 224
    num_channels = 3
    num_classes = 7

    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    # emotion = {0:'愤怒', 1:'厌恶', 2:'恐惧', 3:'高兴', 4:'悲伤', 5:'惊讶', 6: '无表情'}

    # initialize the camera and grab a reference to the raw camera capture
    cap = cv.VideoCapture(video)
    fourcc = cv.VideoWriter_fourcc(*'MPEG')
    out = cv.VideoWriter('output.avi', fourcc, 24.0, (400, 600))

    detection_model = load_detection_model('models/haarcascade_frontalface_default.xml')
    emotion_model = load_emotion_model('models/model.best.hdf5')

    # capture frames from the camera
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.cvtColor(gray, cv.COLOR_GRAY2RGB)
        faces = detect_faces(detection_model, gray)

        for (x, y, w, h) in faces:
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

        # show the frame
        cv.imshow("Frame", frame)

        # if the `q` key was pressed, break from the loop
        key = cv.waitKey(1)
        if key == ord("q"):
            break

    cap.release()
    out.release()
    cv.destroyAllWindows()
