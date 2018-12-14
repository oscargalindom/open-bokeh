"""
Created on Sun Dec  2 21:34:59 2018
Open bokeh, creates a bokeh effect on images

@author: Oscar Galindo and Aleksandr Diamond
@credit: Olac Fuentes for connected components segmentation
"""

import numpy as np
import cv2

from connected_componenets import ConnectedComponents

# thr=0.2
######################################
# local dirs for haar cascades
face_cascade = cv2.CascadeClassifier('/Users/aleksandr/open-bokeh/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('/Users/aleksandr/open-bokeh/lib/python3.6/site-packages/cv2/data/haarcascade_eye.xml')
img = cv2.imread('portrait.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.2, 5)

for (x, y, w, h) in faces:
    # widths are to ignore rectangle
    I = img[y + 2:y + h - 2, x + 2:x + w - 2]  # Change the order to get the actual face Sliced.
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 215, 255), 2)
    cv2.imshow('Face', I)
    # part of facial recognition
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + 10, x:x + 10]

# img = cv2.resize(img, (920,640))
cv2.imshow('Face', I)
cv2.imshow('img', img)
######################################
for (x, y, w, h) in faces:
    cc = ConnectedComponents(I)
    cc.calculate_threshold()
    face = img[y + 2:y + h - 2, x + 2:x + w - 2, :]
    # thr = calculate_threshold(face)
    # print(cc.thr)
    cv2.imshow('Face', face)

    rows = face.shape[0]
    cols = face.shape[1]

    cc_face = ConnectedComponents(face)
    cc_face.calculate_threshold()
    print(cc_face.thr)

    # S, count, sum_pixels = initialize(face)
    # connected_components_segmentation(face, thr)

    cc_face.connected_components_segmentation()
    print('Regions found: ', cc_face.num_regions())
    print('Size 1 regions found: ', cc_face.regions(size=1))
    cc_face.show(mode='mean')
    cc_face.show(mode='rand')

cv2.waitKey(0)
cv2.destroyAllWindows()
