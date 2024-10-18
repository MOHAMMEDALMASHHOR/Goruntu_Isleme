# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 09:55:31 2024

@author: Lenovo
"""

import cv2
img=cv2.imread('Histogram_Sehir.jpeg')
cv2.imshow('image',img)
cv2.waitKey(0)
hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
cv2.imshow("HS Goruntu",hsv)
cv2.waitKey(0)
H = hsv[:,:,1]

cv2.imshow("H Goruntu", H)
cv2.waitKey(0) # After every imshow you must put this code line
#%%
img=cv2.imread('abc.jpg')
cv2.imshow('image',img)
cv2.waitKey(0)
hsv = cv2.cvtColor(img,cv2.COLOR_RGB2XYZ)
cv2.imshow("HS Goruntu",hsv)
cv2.waitKey(0)
H = hsv[:,:,1]

cv2.imshow("H Goruntu", H)
cv2.waitKey(0) # After every imshow you must put this code line
#%%
img=cv2.imread('abc.jpg')
cv2.imshow('image',img)
cv2.waitKey(0)
hsv = cv2.cvtColor(img,cv2.COLOR_RGB2LAB)
cv2.imshow("HS Goruntu",hsv)
cv2.waitKey(0)
H = hsv[:,:,1]

cv2.imshow("H Goruntu", H)
cv2.waitKey(0) # After every imshow you must put this code line
#%%
img=cv2.imread('abc.jpg')
cv2.imshow('image',img)
cv2.waitKey(0)
hsv = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
cv2.imshow("HS Goruntu",hsv)
cv2.waitKey(0)
H = hsv[:,:,1]

cv2.imshow("H Goruntu", H)
cv2.waitKey(0) # After every imshow you must put this code line
#%%
#odev
# import cv2
# img = cv2.imread("doga.jpg")
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# xyz = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)
# lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
# hzl = hsv[:, :, 0]
# xyz_y = xyz[:, :, 2]
# lab_l = lab[:, :, 0]
# merged = cv2.merge([hzl, xyz_y, lab_l])
# cv2.imshow("Merged Image", merged)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#%%
import cv2

def process_input(input_string):
    # Process the input string, for example, extracting values
    # Here I'm assuming you want to extract the number and the letter.
    number = int(input_string[:-1])  # Get the numeric part
    letter = input_string[-1]         # Get the letter part
    # Example transformation based on the input
    # You can modify this to match your requirements
    result = f"(H{number}L{ord(letter) - ord('A') + 1})"
    return result

# Main code
input_string = "26B"  # Example input
output = process_input(input_string)
print(output)  # Output will be (H26L2)

# Now for the image processing part
img = cv2.imread('abc.jpg')
cv2.imshow('image', img)
cv2.waitKey(0)

hsv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
cv2.imshow("HS Goruntu", hsv)
cv2.waitKey(0)

H = hsv[:, :, 1]
cv2.imshow("H Goruntu", H)
cv2.waitKey(0)  # After every imshow you must put this code line
#%%
import cv2
import numpy as np

def process_input(input_string):
    number = int(input_string[:-1])  # Get the numeric part
    letter = input_string[-1]         # Get the letter part
    result = f"(H{number}L{ord(letter) - ord('A') + 1})"
    return result

# Main code
input_string = "26B"  # Example input
output = process_input(input_string)
print(output)  # Output will be (H26L2)

# Load the image
img = cv2.imread('abc.jpg')
cv2.imshow('Original Image', img)
cv2.waitKey(0)

# Convert to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV Image", hsv)
cv2.waitKey(0)

# Convert to LSB (Low-Saturation Brightness)
# Here we'll just create a grayscale image for simplicity
lsb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("LSB Image", lsb)
cv2.waitKey(0)

# Convert to XYZ color space
xyz = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)
cv2.imshow("XYZ Image", xyz)
cv2.waitKey(0)

# Display the final output (HL2)
# For demonstration, let's just show the HSV representation of H and L values
H = hsv[:, :, 0]
L = lsb  # Use the grayscale as L representation
output_image = cv2.merge([H, L, np.zeros_like(H)])  # Merging for display purposes

cv2.imshow("HL2 Image", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#%%
img=cv2.imread('abc.jpg')
cv2.imshow('image',img)
cv2.waitKey(0)
hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)

cv2.imshow("HS Goruntu",hsv)
cv2.waitKey(0)
H = hsv[1:,:,]
lab = cv2.cvtColor(img,cv2.COLOR_RGB2LAB)
l=lab[1,:,:]

xyz = cv2.cvtColor(img,cv2.COLOR_RGB2XYZ)
z=xyz[:,:,1]
photo = img.merge(H,l,z)


cv2.imshow("H Goruntu", H)
cv2.waitKey(0) # After every imshow you must put this code line
cv2.imshow("Sorunu photoso",photo)
cv2.waitKey(0)
'''
Certainly! Let's break down the code and understand what each conversion does. These lines of code are transforming an image from its original *BGR* color space (which is how OpenCV loads images by default) into different color spaces. Each color space represents colors in a unique way. Here's the breakdown:

1. **HSV (Hue, Saturation, Value)**:
   - `hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)`
   - In the HSV color space:
     - *Hue (H)*: Represents the color type (ranging from 0 to 360 degrees on the color wheel). In OpenCV, it's normalized to 0-180.
     - *Saturation (S)*: Indicates the intensity or purity of the color (ranging from 0 to 255).
     - *Value (V)*: Reflects the brightness of the color (ranging from 0 to 255).
   - HSV is commonly used for color-based object tracking and separation of color information from intensity.

2. **XYZ (CIE 1931 XYZ)**:
   - `xyz = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)`
   - The XYZ color space represents color based on human vision's perception of brightness and chromaticity:
     - *X*: Mix of RGB values weighted toward the human eye's sensitivity to red.
     - *Y*: Luminance (brightness) of the image.
     - *Z*: Mix of RGB values weighted toward the human eye's sensitivity to blue.
   - XYZ is useful in scientific and perceptual-based applications.

3. **Lab (CIE L\a\*b\)**:
   - `lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)`
   - Lab is designed to be perceptually uniform, meaning that a change of the same amount in a color value should produce a perceived change of similar visual importance:
     - *L\**: Lightness (0-100), representing brightness.
     - *a\**: The green-red component (negative values for green, positive for red).
     - *b\**: The blue-yellow component (negative for blue, positive for yellow).
   - Lab is widely used for color correction and improving perceived color accuracy.

Remember, each color space serves specific purposes in image processing and computer vision. Depending on your task, you can choose the most suitable one! 😊 Let me know if you need further assistance or have any other questions! 🌟
'''
#%%
#Histogram

import cv2 
import numpy as np
#import matplotlib.pyplot as plt ##this is a libaray for advanced levels

img = cv2.imread('Histogram_Lineer.jpeg',0)
img = cv2.resize(img, (480,320))

cv2.imshow("Orjinal Gorunto",img)
cv2.waitKey(0)

mn = np.min(img)

mx = np.max(img)

im1= (img-mn)/(mx-mn)*255

im2 = im1.astype(np.uint8)
cv2.imshow("Contrast Germe-1",im2)
cv2.waitKey(0)
#%%
#Gamma
import cv2 
import numpy as np
#import matplotlib.pyplot as plt ##this is a libaray for advanced levels

img = cv2.imread('Histogram_Lineer.jpeg',0)
img = cv2.resize(img, (480,320))

cv2.imshow("Orjinal Gorunto",img)
cv2.waitKey(0)

mn = np.min(img)

mx = np.max(img)

im1= (img-mn)/(mx-mn)*255

im2 = im1.astype(np.uint8)
cv2.imshow("Contrast Germe-1",im2)
cv2.waitKey(0)

gamma = 2 # 1,1.5,6,.......
gamma_corrected = np.array(255*(img / 255)**gamma, dtype = 'uint8')

cv2.imshow('log_image-3',gamma_corrected)
cv2.waitKey(0)
#%%
#ChatGPT log code
import cv2 
import numpy as np

# Load and resize the image
img = cv2.imread("Histogram_Lineer.jpeg", 0)
img = cv2.resize(img, (480, 320))

# Display the original image
cv2.imshow("Original Image", img)
cv2.waitKey(0)

# Min and max values
mn = np.min(img)
mx = np.max(img)

# Linear contrast stretching
im1 = (img - mn) / (mx - mn) * 255
im2 = im1.astype(np.uint8)
cv2.imshow("Contrast Stretching", im2)
cv2.waitKey(0)

# Logarithmic transformation
# Adding a small value to avoid log(0)
c = 255 / np.log(1 + mx)  # Scale factor
log_transformed = c * np.log(1 + img)

# Convert to uint8
log_transformed = log_transformed.astype(np.uint8)
cv2.imshow('Log Transformed Image', log_transformed)
cv2.waitKey(0)

# Close all windows
#cv2.destroyAllWindows()
#%%
import cv2

img = cv2.imread('C:/Users/Lenovo/Desktop/Goruntu_Isleme/Week_2/Histogram_Sehir.jpeg')

if img is not None:
    cv2.imshow('image', img)
    cv2.waitKey(0)
    equ = cv2.equalizeHist(img)

    cv2.imshow('img', equ)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Görüntü yüklenemedi.")
#%%
import cv2

# Görüntüyü yükle
img = cv2.imread('C:/Users/Lenovo/Desktop/Goruntu_Isleme/Week_2/Histogram_Sehir.jpeg')

if img is not None:
    # Görüntüyü grayscale formatına dönüştür
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Histogram eşitlemesini uygula
    equalized_img = cv2.equalizeHist(gray_img)
    
    # Sonucu göster
    cv2.imshow('Equalized Image', equalized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Görüntü yüklenemedi.")
#%%
import cv2
import numpy as np

































