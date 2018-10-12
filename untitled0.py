# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 15:01:21 2018

@author: DerMin
"""
#load image
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2
image1 = np.fromfile("BABOON.raw", dtype='uint8', sep="")
image2 = np.fromfile("lena.raw", dtype='uint8', sep="")
image_new_1 = image1.reshape(512,512) 
image_new_2 = image2.reshape(512 , 512)  
im_1 = Image.fromarray(image_new_1)
im_2 = Image.fromarray(image_new_2)
im_1.save("out_1.jpeg")
im_2.save('out_2.jpeg')

#use sobel mask
img_gray_1 = cv2.imread('out_1.jpeg', cv2.IMREAD_GRAYSCALE)
img_gray_2 = cv2.imread('out_2.jpeg', cv2.IMREAD_GRAYSCALE)
sobelx_1 = cv2.Sobel(img_gray_1 ,cv2.CV_16S,1,0,ksize=5)
sobely_1 = cv2.Sobel(img_gray_1,cv2.CV_16S,0,1)
sobelx_2 = cv2.Sobel(img_gray_2 ,cv2.CV_16S,1,0,ksize=5)
sobely_2 = cv2.Sobel(img_gray_2,cv2.CV_16S,0,1)
plt.imshow(sobelx_1 , cmap = 'gray')
plt.imshow(sobely_1 , cmap = 'gray')
plt.imshow(sobelx_2 , cmap = 'gray')
plt.imshow(sobely_2 , cmap = 'gray')

#use Laplacian mask

gray_lap_1 = cv2.Laplacian(img_gray_1,cv2.CV_16S,ksize = 5)
gray_lap_2 = cv2.Laplacian(img_gray_2,cv2.CV_16S,ksize = 5)
plt.imshow(gray_lap_1 , cmap = 'gray')
plt.imshow(gray_lap_2 , cmap = 'gray')


#use average mask

blur_1 = cv2.blur(img_gray_1,(5,5))
blur_2 = cv2.blur(img_gray_2,(5,5))
plt.imshow(blur_1 , cmap = 'gray')
plt.imshow(blur_2 , cmap = 'gray')

#Gaussian mask

gauss_1 = cv2.GaussianBlur(img_gray_1 , (5,5) , 0)
gauss_2 = cv2.GaussianBlur(img_gray_2 , (5,5) , 0)
plt.imshow(gauss_1 , cmap = 'gray')
plt.imshow(gauss_2 , cmap = 'gray')


#Generate Gaussian noise

gauss = np.random.normal(0 , 10 , (512 , 512) )
noisy_1 = gauss + img_gray_1
noisy_2 = gauss + img_gray_2
plt.imshow(noisy_1 , cmap = 'gray')
plt.imshow(noisy_2 , cmap = 'gray')


#remove noise by gaussian filter
noisy_1 = cv2.GaussianBlur(img_gray_1 , (5,5) , 0)
noisy_2 = cv2.GaussianBlur(img_gray_2 , (5,5) , 0)
plt.imshow(noisy_1 , cmap = 'gray')
plt.imshow(noisy_2 , cmap = 'gray')

# 100 Gaussian
total_1 = 0
total_2 = 0
for i in range(100):
    gauss = np.random.normal(0 , 10 , (512 , 512) )
    noisy_1 = gauss + img_gray_1
    noisy_2 = gauss + img_gray_2
    total_1 += noisy_1
    total_2 += noisy_2
de_noise_1 = total_1/100
de_noise_2 = total_2/100
plt.imshow(de_noise_1 , cmap = 'gray')
plt.imshow(de_noise_2 , cmap = 'gray')






