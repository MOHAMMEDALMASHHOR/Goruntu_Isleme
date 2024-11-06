# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 09:45:39 2024

@author: Lenovo
"""

import numpy as np
import cv2 
from matplotlib import pyplot as plt

img1 = cv2.imread('FingerPrint.jpg')
img1 = cv2.resize(img1, (1000,550))
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) 

ret,thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO_INV)

titles = ['Original Image', 'BINARY', 'BINARY_INV','TRUNC','TOZERO','TOZERO_INV']

images = [gray,thresh1,thresh2,thresh3,thresh4,thresh5]

for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()
    