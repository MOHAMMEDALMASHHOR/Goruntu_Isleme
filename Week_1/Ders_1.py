# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 21:17:28 2024

@author: Lenovo
"""
#%%
import cv2

# Read the PNG image
image = cv2.imread('abc.png')

# Save it as JPG
cv2.imwrite('abc.jpg', image)
#%%
import cv2
img = cv2.imread("abc.jpg")
img1 = cv2.imread("abc.jpg",0)
cv2.imshow("Orjinal", img)
cv2.imshow("Gri Görüntü", img1)
cv2.waitKey(0)