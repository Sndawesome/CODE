# -*- coding: utf-8 -*-
"""
Created on Mon May 21 10:29:53 2018

@author: Sandesh
"""
from tkinter import *
from PIL import Image, ImageTk
from main import *
from threading import Thread
import webbrowser
import cv2
from gender_model import *
from keras.models import load_model

model_emotion=load_jason_model('bigger_model')
model_gender=model = load_gender_model()
def Predict():
    return emotion_model_on_live_cam(model_emotion,1)
def predict_gender():
    return gender_model_on_live_cam(model_gender,1)
root=Tk()
root.title("Emotion recogmizer")
button1 = Button(root, text='Emotion Recogniser', command=Predict)
button1.grid(row=2,column=2)
button2 = Button(root, text='Gender Recogniser', command=predict_gender)

button2.grid(row=0, column=2)
root.mainloop()
