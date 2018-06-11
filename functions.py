# -*- coding: utf-8 -*-
"""
Created on Mon May 14 11:04:45 2018

@author: Sandesh
"""
import numpy as np
import cv2
import pandas as pd
from PIL import Image
SIZE_FACE=48
EMOTIONS = ['angry', 'disgusted', 'fearful',
            'happy', 'sad', 'surprised', 'neutral']

cascade_classifier = cv2.CascadeClassifier("C:\opencv\sources\data\haarcascades\haarcascade_frontalface_alt2.xml")
def data_to_image(data):
    data_image = np.fromstring(
        str(data), dtype=np.uint8, sep=' ').reshape((SIZE_FACE, SIZE_FACE))
    data_image = Image.fromarray(data_image).convert('RGB')
    data_image = np.array(data_image)[:, :, ::-1].copy()
    #data_image = format_image(data_image)
    return data_image
def format_image(image):
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    gray_border = np.zeros((150, 150), np.uint8)
    gray_border[:, :] = 200
    gray_border[
        int((150 / 2) - (SIZE_FACE / 2)): int((150 / 2) + (SIZE_FACE / 2)),
        int((150 / 2) - (SIZE_FACE / 2)): int((150 / 2) + (SIZE_FACE / 2))
    ] = image
    image = gray_border
    faces = cascade_classifier.detectMultiScale(
        image,
        scaleFactor=1.3,
        minNeighbors=5
    )
    #  None is we don't found an image
    if not len(faces) > 0:
        return None
    max_area_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face
    # Chop image to face
    face = max_area_face
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
    # Resize image to network size

    try:
        image = cv2.resize(image, (SIZE_FACE, SIZE_FACE),
                           interpolation=cv2.INTER_CUBIC) / 255.
    except Exception:
        print("[+] Problem during resize")
        return None
    return image

def emotion_to_vec(x):
    d = np.zeros(len(EMOTIONS))
    d[x] = 1.0
    return d
