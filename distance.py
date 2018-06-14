# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 17:25:18 2018

@author: Sandesh
"""

import cv2 
import numpy as np
from statistics import mean
import time
import serial

face_cascade = cv2.CascadeClassifier("C:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml")
width=16
KNOWN_DISTANCE = 32
def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth 

def hist(image):
    hist,bins = np.histogram(image.flatten(),256,[0,256])
     
    cdf = hist.cumsum()
    #cdf_normalized = cdf * hist.max()/ cdf.max()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    img2 = cdf[image]
    return img2

focal_length=290*KNOWN_DISTANCE/width
def distance(cap):
  arr=[]
  count=0
  while count<7:
    ret,img=cap.read()
    img2=hist(img)  
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.4,8)
    if len(faces)>0:
        count+=1
    for (x,y,w,h) in faces:
      cv2.rectangle(img,(x,y),(x+w,y+h),(110,110,110),0)
      font=cv2.FONT_HERSHEY_COMPLEX
      arr.append(distance_to_camera(width,focal_length,w))
      cv2.putText(img,str(int(distance_to_camera(width,focal_length,w))),(x,y),font,1,(255,255,255),1,cv2.LINE_AA)
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k==27:
      cap.release()
      break
  try:
      return int(mean(arr))
  except:
      return

if __name__=='__main__':
    sleep=10
    prev_time=time.time()-sleep
    current_time=time.time()
    cap = cv2.VideoCapture(0)
    #ser = serial.Serial('COM3', 9600,timeout=0)
    while cap.isOpened() :
        if((int(current_time)-int(prev_time))>=sleep):
            e=distance(cap)
            print(e)
            #ser.write(str(e).encode())
            time.sleep(1)   
            current_time=time.time()
            prev_time=current_time 
        else:
          ret,img=cap.read()
          cv2.imshow('img',img)
          k = cv2.waitKey(30) & 0xff
          if k==27:
              break 
          current_time=time.time()
    #ser.close()   
    cap.release()
    cv2.destroyAllWindows()
