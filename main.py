# -*- coding: utf-8 -*-
"""
Created on Mon May 14 10:47:16 2018

@author: Sandesh
"""

import pandas as pd
import numpy as np
import cv2
from functions import *
import os
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json
#import dlib
from statistics import mode

def hist(image):
    hist,bins = np.histogram(image.flatten(),256,[0,256])
     
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    
    '''plt.plot(cdf_normalized, color = 'b')
    plt.hist(img.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()'''
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    img2 = cdf[image]
    return img2
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def train_model():   
 classifier = Sequential()

# Step 1 - Convolution
 classifier.add(Convolution2D(64, 3, 3, input_shape = (48, 48,3), activation = 'relu'))

# Step 2 - Pooling
 classifier.add(MaxPooling2D(pool_size = (2, 2)))
 #classifier.add(Convolution2D(32, 4, 4, activation = 'relu'))

# Adding a second convolutional layer
 classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
 classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
 classifier.add(Flatten())

# Step 4 - Full connection
 classifier.add(Dense(output_dim = 512, activation = 'relu'))
 #classifier.add(Dense(output_dim=128,activation='relu'))
 classifier.add(Dense(output_dim=28,activation='relu'))
 classifier.add(Dense(output_dim = 7, activation = 'sigmoid'))

# Compiling the CNN
 classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy',precision])

 from keras.preprocessing.image import ImageDataGenerator
 train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

 test_datagen = ImageDataGenerator(rescale = 1./255)

 training_set = train_datagen.flow_from_directory('imagedata/training_set',
                                                 target_size = (48, 48),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
 test_set = test_datagen.flow_from_directory('imagedata/test_set',
                                            target_size = (48, 48),
                                            batch_size = 32,
                                            class_mode = 'categorical')


 classifier.fit_generator(training_set,
                         samples_per_epoch = 28658,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 8523)
 classifier.summary()
 return classifier

EMOTIONS = ['angry', 'disgusted', 'fearful',
            'happy', 'sad', 'surprised', 'neutral']
#Prediction
#class_labels = {v: k for k, v in training_set.class_indices.items()}
def predict(img,classifier):
 img = cv2.resize(img,(48,48))
 img = np.expand_dims(img,axis=0)
 if(np.max(img)>1):
    img = img/255.0
 
 prediction = classifier.predict_classes(img)
 return prediction

 
 #Saving model
def save_model(classifier,name='model_1'):
 model_json = classifier.to_json()
 with open("%s.json"%name, "w") as json_file:
    json_file.write(model_json)
 # serialize weights to HDF5
 classifier.save_weights("%s.h5"%name)
 print("Saved model to disk")

#Loading model
def load_jason_model(name='model_1'):
 json_file = open('%s.json'%name, 'r')
 loaded_model_json = json_file.read()
 json_file.close()
 loaded_model = model_from_json(loaded_model_json)
 # load weights into new model
 loaded_model.load_weights("%s.h5"%name)
 print("Loaded model from disk")
 loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 return loaded_model

#detector = dlib.get_frontal_face_detector()
def emotion_model_on_live_cam(classifier,cam=0):
 face_cascade = cv2.CascadeClassifier("C:\opencv\sources\data\haarcascades\haarcascade_frontalface_alt2.xml")
 #detector = dlib.get_frontal_face_detector()
 cap = cv2.VideoCapture(0)
 cap.set(cv2.CAP_PROP_FPS, 1200)
 count=0
 arr=[]

 while cap.isOpened() :
    ret, img = cap.read()
    img2=hist(img)
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(110,110,110),1)
        image=img[y:y+h,x:x+w]
        image=cv2.resize(image,(48,48))
        image=np.expand_dims(image,axis=0)
        if(np.max(image)>1):
            image = image/255.0
        prediction = classifier.predict(image)
        #prediction=softmax(prediction)
        c=classifier.predict_classes(image)
        
        arr.append(c[0])
        #eyes = eye_cascade.detectMultiScale(roi_gray)
        font=cv2.FONT_HERSHEY_COMPLEX
        
        cv2.putText(img,EMOTIONS[c[0]],(x,y),font,1,(255,255,255),1,cv2.LINE_AA)
        #cv2.putText(img,EMOTIONS[p.argmax()],(x,y+h),font,1,(255,255,255),1,cv2.LINE_AA)

        if c[0] is not None:
            for index, emotion in enumerate(EMOTIONS):
             cv2.putText(img, emotion, (10, index * 20 + 20),
                               cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
             cv2.rectangle(img, (130, index * 20 + 10), (130 +
                         int(prediction[0][index] * 100), (index + 1) * 20 + 4), (255, 0, 0), -1)
        count+=1
    cv2.imshow('img',img)
            
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
 cap.release()
 cv2.destroyAllWindows()
 print(count) 
 print(arr)
 #print(EMOTIONS[mode(arr)])
 

if __name__ == "__main__":
    print('What do you want: train_model/load_model ')
    a=input()
    if a=='train_model':
        print('Save model name by: ')
        name=input()
        print('Training....')
        model=train_model()
        save_model(model,name)
    if a=='load_model':
        print('Model name: ')
        m=input()
        while True:
         try:
          model=load_jason_model(m)
          break
         except:
            print('Type saved model name:')
            m=input()
    print('Do you want to run it on live cam: Y/N ')
    b=input()
    while True:
        if(b=='Y'):
            emotion_model_on_live_cam(model)
            break
        if(b=='N'):
            print('Run on camera next time. ')
            break
        else:
            print('Enter Y/N :')
            b=input()


def softmax(a):
    expA=np.exp(a)
    return expA/expA.sum(axis=1,keepdims=True)








