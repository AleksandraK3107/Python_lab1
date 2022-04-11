# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 20:43:39 2022

@author: 48535
"""

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from imutils import face_utils
import pyautogui as pyag
import imutils
import dlib
import mouse


print("1. Show your palm to detect your face."
      "\n2. Move your head to move the mouse. :)"
      "\n3. Show a thumbs up to left-click."
      "\n4. Show a thumbs down to right-click."
      "\n5. Show a call me to scroll up."
      "\n6. Show a fist to scroll down."
      "\n7. Press ESC to exit.")

# Defining the directions of the nose tip shifting in relation to its starting point
def direction(nose_point, anchor_point, w, h, multiple=1):
    nx, ny = nose_point
    x, y = anchor_point

    if nx > x + multiple * w:
        return 'RIGHT'
    elif nx < x - multiple * w:
        return 'LEFT'

    if ny > y + multiple * h:
        return 'DOWN'
    elif ny < y - multiple * h:
        return 'UP'

    return 'NO MOTION DETECTED'

# Variables
INPUT_MODE = False
SCROLL_MODE = False
ANCHOR_POINT = (0, 0)
WHITE_COLOR = (255, 255, 255)
YELLOW_COLOR = (0, 255, 255)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (255, 0, 0)
BLACK_COLOR = (0, 0, 0)

# Face detecting using library dlib
shape_predictor = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


# Detection of hand gestures using the model
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

model = load_model('mp_hand_gesture')

# Loading the names of hand gestures
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)


cap = cv2.VideoCapture(0)
resolution_w = 1366
resolution_h = 768
cam_w = 640
cam_h = 480
unit_w = resolution_w / cam_w
unit_h = resolution_h / cam_h

while True:
    _, frame = cap.read()
    
    a, b, c = frame.shape
    
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=cam_w, height=cam_h)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
     # Detecting hand gestures
    result = hands.process(framergb)

    className = ''

    # Detecting a face in gray frame
    rects = detector(gray, 0)

    
    if len(rects) > 0:
        rect = rects[0]
    else:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        continue

    # Defining landmarks on face
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    
    mouth = shape[mStart:mEnd]
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    nose = shape[nStart:nEnd]
    
    temp = leftEye
    leftEye = rightEye
    rightEye = temp


    nose_point = (nose[3, 0], nose[3, 1])
    
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * a)
                lmy = int(lm.y * b)

                landmarks.append([lmx, lmy])

            # Drawing landmarks
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predicting hand gesture
            prediction = model.predict([landmarks])
            classID = np.argmax(prediction)
            className = classNames[classID]

        
    # CLICKING
    if className == "thumbs down":
        mouse.click('right')


    if className == "thumbs up":
        mouse.click('left')

    if className == "stop":
        ANCHOR_POINT = nose_point
        INPUT_MODE = not INPUT_MODE
        
    if INPUT_MODE:
        cv2.putText(frame, "MOUSE MOVEMENT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)
        x, y = ANCHOR_POINT
        nx, ny = nose_point
        w, h = 60, 35
        multiple = 1
        cv2.rectangle(frame, (x - w, y - h), (x + w, y + h), GREEN_COLOR, 2)
        cv2.line(frame, ANCHOR_POINT, nose_point, BLUE_COLOR, 2)

        dir = direction(nose_point, ANCHOR_POINT, w, h)
        cv2.putText(frame, dir.upper(), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)
        drag = 18
        if dir == 'RIGHT':
            pyag.moveRel(drag, 0)
        elif dir == 'LEFT':
            pyag.moveRel(-drag, 0)
        elif dir == 'UP':
            pyag.moveRel(0, -drag)
        elif dir == 'DOWN':
            pyag.moveRel(0, drag)
                
                    
    # Scroll
    if className == "call me":
        pyag.scroll(20)
    if className == "fist":
        pyag.scroll(-20)
        cv2.putText(frame, 'SCROLL!', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)


    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF


    if key == 27 or 0xFF == ord('q'):
        break



cap.release()

cv2.destroyAllWindows()