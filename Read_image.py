# -*- coding: utf-8 -*-
"""
Created on Wed May  9 15:25:21 2018

@author: Sandesh
"""

import cv2
import matplotlib.pyplot as plt

def read_image(path,value=0):
    img=cv2.imread(path,value)
    cv2.waitKey(0)
    return img

def show_image(np_array):
    try:
        print('Exit with esc')
        cv2.imshow('img',np_array)
        k=cv2.waitKey(0) & 0xff
        if k == 27:
          cv2.destroyAllWindows()
    except:
        print("No data")
        pass
    
def write_image(name,np_array):
    cv2.imwrite(name,np_array)

        
