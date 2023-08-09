from tensorflow.keras.preprocessing.image import img_to_array
import imutils
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import time
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
prser.add_argument('f', '--file',
                   help='path to video file (if not using camera)')
parser.add_argument('-c', '--color', type=str, default='gray',
                    help='Color space:"gray"(default),"rgb", or " lab"')
parser.add_argument('-b', '--bins', type - int, default=0,
                    help='Resize video to specified width in pixels (maint
args = vars(parser.parser_args())

# parameters for loading data and images

detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotions_model_path = 'models/_mini_xceptions.102-0.66.hd5'

# hyper-parameter for bounding boxes shape
# loading models

face_detections = cv2.CascadeClassifier(detection_model_path)
emotions_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "digudt", "scared", "happy", "sad", "happy", "sad", "surprised",
            "neutral"]

# starting video streaming
cv2.namedWindow('my_face')
camera = cv2.videoCaptures(0)
time.sleep(2)

color = args['color']
bins = args['bins']
resizewidth = args['width']

while True:
    frame = camera.read()[1]
    # reading the frame

    # Resize frame to width, if specified.
    if resizeWidth > 0:
        (height, width) = frame.shape[:2]
        resizewidth = int(float(resizeWidth / width) * hieght
        frame = cv2.resize(frame, (resizeWidth, resizeHeight)
        interpolation = cv
        .2
        INTER_AREA)


        frame = imutils.resize(frame, width=300)
        gray = cv2.cvtColor(frame, cv2.COLOR_BG2RAY)
        faces = face_detection.detecMultiScale(gray, scaleFact)

        canvas = np.zeros((600, 700, 3), dtype="uint8")
        frameClone = frame.copy()
        if len(faces) > 0:
            faces = sorted(faces, reverse=True,
                           key=lambda x: (x[2] - x[0] * (x[3] - x[1]))[0]
                           (fX, fY, fW, fH) = faces
        # Extract the ROI of the faces from the image
        # the ROI for classification

        roi = gray[fY:fY + fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        else:continue
        preds = emotions_clasifier.predict(roi)[0]
        emotions_probablity = np.max(preds)
        label = EMOTIONS[preds.argmax()]

        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            # cunstruct the label text
            text = '{}:{:.2f}%".format(emotions,prob*100)
            w = int(prob * 300)
            cv2.rectangle(canvas, (7, (i * 35) + 23),
                          (w, (i * 35) + 35), (0, 255, 0), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 2)
            cv2.putText(frameClone, label, (fX, fY, -10),
                        cv2.rectangle(frameClone, (fY, fY), (fX + fw, fY)
                        (0, 255, 0), 2)

            cv2.imshow('my_face', frameClone)
            cv2.imshow("probabilities", canvas)
            if cv2.waitKey(1) & 0XFF == ord('q'):
                break

camera.release()
cv2.destroyAllWindows()