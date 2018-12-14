"""
Created on Sun Dec  2 21:34:59 2018
Open bokeh, creates a bokeh effect on images

@author: Oscar Galindo and Aleksandr Diamond
@credit: Olac Fuentes for connected components segmentation
"""

import numpy as np
import cv2


def find(i):
    if S[i] < 0:
        return i
    s = find(S[i])
    S[i] = s  # Path compression
    return s


def union(i, j, thr):
    # Joins two regions if they are similar
    # Keeps track of size and mean color of regions
    ri = find(i)
    rj = find(j)
    if ri != rj:
        d = sum_pixels[ri, :] / count[ri] - sum_pixels[rj, :] / count[rj]
        diff = np.sqrt(np.sum(d * d))
        if diff < thr:
            S[rj] = ri
            count[ri] += count[rj]
            count[rj] = 0
            sum_pixels[ri, :] += sum_pixels[rj, :]
            sum_pixels[rj, :] = 0


def regular_union(i, j, thr):
    # Joins two regions if they are similar
    # Keeps track of size and mean color of regions
    ri = find(i)
    rj = find(j)
    if ri != rj:
        S[rj] = ri
        count[ri] += count[rj]
        count[rj] = 0
        sum_pixels[ri, :] += sum_pixels[rj, :]
        sum_pixels[rj, :] = 0


def initialize(I):
    rows = I.shape[0]
    cols = I.shape[1]
    S = np.zeros(rows * cols).astype(np.int) - 1
    count = np.ones(rows * cols).astype(np.int)
    sum_pixels = np.copy(I).reshape(rows * cols, 3)
    return S, count, sum_pixels


def connected_components_segmentation(I, thr):
    rows = I.shape[0]
    cols = I.shape[1]
    for p in range(S.shape[0]):
        if p % cols < cols - 1:  # If p is not in the last column
            union(p, p + 1, thr)  # p+1 is the pixel to the right of p
        if p // cols < rows - 1:  # If p is not in the last row
            union(p, p + cols, thr)  # p+cols is the pixel to below p


def calculate_threshold(I):
    std = [np.std(I[:, :, 0]), np.std(I[:, :, 1]), np.std(I[:, :, 2])]
    return std[np.argmax(std)]


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
    face = img[y + 2:y + h - 2, x + 2:x + w - 2, :]
    thr = calculate_threshold(face)
    print(thr)
    cv2.imshow('Face', face)

    rows = face.shape[0]
    cols = face.shape[1]
    S, count, sum_pixels = initialize(face)
    connected_components_segmentation(face, thr)

    print('Regions found: ', np.sum(S == -1))
    print('Size 1 regions found: ', np.sum(count == 1))

    rand_cm = np.random.random_sample((rows * cols, 3))
    seg_im_mean = np.zeros((rows, cols, 3))
    seg_im_rand = np.zeros((rows, cols, 3))
    for r in range(rows - 1):
        for c in range(cols - 1):
            f = find(r * cols + c)
            seg_im_mean[r, c, :] = sum_pixels[f, :] / count[f]
            seg_im_rand[r, c, :] = rand_cm[f, :]

    cv2.imshow('Segmentation 1 - using mean colors', seg_im_mean)
    cv2.imshow('Segmentation 2 - using random colors', seg_im_rand)
cv2.waitKey(0)
cv2.destroyAllWindows()
