# -*- coding: utf-8 -*-
"""
Created on Thu May 24 11:00:30 2018

@author: Sandesh
"""

import cv2
import os
from Read_image import *
import numpy as np
'''files=os.listdir('female')
for file in files:
    image=read_image('female/'+file,1)
    image=cv2.resize(image,(64,48))
    cv2.imwrite('female/'+file,image)'''
    
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json
from keras.models import load_model


'''classifier = Sequential()
classifier.add(Convolution2D(64, 3, 3, input_shape = (48, 64,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(32, 4, 4, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())

classifier.add(Dense(output_dim=128,activation='relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('training_set',
                                                 target_size = (48, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_set = test_datagen.flow_from_directory('test_set',
                                            target_size = (48, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


classifier.fit_generator(training_set,
                         samples_per_epoch = 2800,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 146
                         )
'''
def save_model(classifier,name='model_1'):
 model_json = classifier.to_json()
 with open("%s.json"%name, "w") as json_file:
    json_file.write(model_json)
 # serialize weights to HDF5
 classifier.save_weights("%s.h5"%name)
 print("Saved model to disk")

GENDER=['Female','Male']

 
def get_labels(dataset_name):
    if dataset_name == 'fer2013':
        return {0:'angry',1:'disgust',2:'fear',3:'happy',
                4:'sad',5:'surprise',6:'neutral'}
    elif dataset_name == 'imdb':
        return {0:'woman', 1:'man'}
    elif dataset_name == 'KDEF':
        return {0:'AN', 1:'DI', 2:'AF', 3:'HA', 4:'SA', 5:'SU', 6:'NE'}
    else:
        raise Exception('Invalid dataset name')
def load_detection_model(model_path):
    detection_model = cv2.CascadeClassifier(model_path)
    return detection_model


labels = get_labels('imdb')
offsets = (30, 60)
def load_gender_model(model_filename = 'trained_models/gender_models/gender_mini_XCEPTION.21-0.95.hdf5'):
 model = load_model(model_filename, compile=False)
 print('Model loaded from disk.')
 return model

model=load_gender_model()

def gender_model_on_live_cam(classifier,cam=0):
 face_cascade = cv2.CascadeClassifier("C:\opencv\sources\data\haarcascades\haarcascade_frontalface_alt2.xml")
 cap = cv2.VideoCapture(cam)
 count=0

 while cap.isOpened():
    ret, img = cap.read()
    img2=hist(img)
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(110,110,110),1)
        image=gray[y:y+h,x:x+w]
        image=cv2.resize(image,(64,64))
        image=image.reshape((64,64,1))
        image=np.expand_dims(image,axis=0)
        if(np.max(image)>1):
            image = image/255.0
        prediction = classifier.predict(image)
        #print(prediction)
        font=cv2.FONT_HERSHEY_COMPLEX
        
        cv2.putText(img,GENDER[np.argmax(prediction)],(x,y),font,1,(255,255,255),1,cv2.LINE_AA)
        '''if c[0] is not None:
            for index, emotion in enumerate(EMOTIONS):
             cv2.putText(img, emotion, (10, index * 20 + 20),
                               cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
             cv2.rectangle(img, (130, index * 20 + 10), (130 +
                         int(prediction[0][index] * 100), (index + 1) * 20 + 4), (255, 0, 0), -1)'''
    count+=1
    cv2.imshow('img',img)
            
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
 print(count)    
 cap.release()
 cv2.destroyAllWindows()


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








