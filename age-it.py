#OpenCV module
import cv2
#os module for reading training data directories and paths
import os
#numpy to convert python lists to numpy arrays as it is needed by OpenCV face recognizers
import numpy as np

from matplotlib import pyplot as plt

#importing time library for speed comparisons of both classifiers
import time 

#cargar imagen
obama_image = cv2.imread("C:/Users/PENTA/Desktop/proyectos/age-it/imagenes/obama-trump.jpg")

#convert the test image to gray image as opencv face detector expects gray images 
gray_img = cv2.cvtColor(obama_image, cv2.COLOR_BGR2GRAY)

#load cascade classifier training file for haarcascade 
haar_face_cascade = cv2.CascadeClassifier('C:/Users/PENTA/Desktop/proyectos/age-it-env/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml')

#let's detect multiscale (some images may be closer to camera than others) images 
faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=1)

#print the number of faces found 
print('Faces found: ', len(faces))

faces

#go over list of faces and draw them as rectangles on original colored 
for(x,y,w,h) in faces:
    print(x,y,w,h)
    cv2.rectangle(gray_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

while(True):
    cv2.imshow('hola',gray_img)
    k = cv2.waitKey(0)
    print(k)
    if k == 27: 
        cv2.destroyAllWindows()
        break