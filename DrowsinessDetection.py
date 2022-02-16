import cv2
import os
import keras
from keras.models import load_model
import numpy as np
# from pygame import mixer
import time

face = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haarcascade/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haarcascade/haarcascade_righteye_2splits.xml')

eye_labels = ['closed', 'open']
yawn_labels = ['no_yawn', 'yawn']

eye_model = load_model('models/cnnEyes.h5')
yawn_model = load_model('models/cnnYawn.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
rpred = [99]
lpred = [99]
yawn_pred = [0]

while(True):
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(
        gray,    minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height-60), (250, height),
                  (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 1)

        # Yawn Predictor
        cut = frame[y:y+h, x:x+w]
        cut = cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY)
        cut = cv2.resize(cut, (24, 24))
        cut = cut/255
        cut = cut.reshape(24, 24, -1)
        cut = np.expand_dims(cut, axis=0)
        yawn_pred = np.argmax(yawn_model.predict(cut), axis=-1)
        # print('Prediction: ', yawn_labels[yawn_pred[0]])
        break

    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y+h, x:x+w]
        count = count+1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye/255
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)
        # rpred = model.predict(r_eye)
        # rpred_class = keras.np_utils.probas_to_classes(rpred)
        rpred = np.argmax(eye_model.predict(r_eye), axis=-1)
        if(rpred[0] == 1):
            lbl = 'Open'
        if(rpred[0] == 0):
            lbl = 'Closed'
        break

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y+h, x:x+w]
        count = count+1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye/255
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)
        # lpred = model.predict(l_eye)
        # lpred_class = keras.np_utils.probas_to_classes(lpred)
        lpred = np.argmax(eye_model.predict(l_eye), axis=-1)
        if(lpred[0] == 1):
            lbl = 'Open'
        if(lpred[0] == 0):
            lbl = 'Closed'
        break

    # For eyes
    if(rpred[0] == 0 and lpred[0] == 0):
        score = score+1
        cv2.putText(frame, "Eyes Closed", (10, height-20), font,
                    1, (255, 255, 255), 1, cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    else:
        score = score-1
        cv2.putText(frame, "Eyes Open", (10, height-20), font,
                    1, (255, 255, 255), 1, cv2.LINE_AA)

    # For yawn
    if(yawn_pred[0] == 0):
        cv2.putText(frame, "No Yawn", (10, height-40), font,
                    1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score = score+1
        cv2.putText(frame, "Yawn", (10, height-40), font,
                    1, (255, 255, 255), 1, cv2.LINE_AA)

    if(score < 0):
        score = 0
    cv2.putText(frame, 'Score:'+str(score), (150, height-20),
                font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    if(score > 20):
        # person is feeling sleepy so we beep the alarm
        # cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        # try:
        #     sound.play()

        # except:  # isplaying = False
        #     pass
        if(thicc < 16):
            thicc = thicc+2
        else:
            thicc = thicc-2
            if(thicc < 2):
                thicc = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
