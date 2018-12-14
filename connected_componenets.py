import numpy as np
import cv2


# from project import S, sum_pixels, count


class ConnectedComponents:
    def __init__(self, I):
        self.I = I
        self.rows = I.shape[0]
        self.cols = I.shape[1]
        self.S = np.zeros(self.rows * self.cols).astype(np.int) - 1
        self.count = np.ones(self.rows * self.cols).astype(np.int)
        self.sum_pixels = np.copy(I).reshape(self.rows * self.cols, 3)

    def find(self, i):
        if self.S[i] < 0:
            return i
        s = self.find(self.S[i])
        self.S[i] = s  # Path compression
        return s

    def union(self, i, j):
        # Joins two regions if they are similar
        # Keeps track of size and mean color of regions
        ri = self.find(i)
        rj = self.find(j)
        if ri != rj:
            d = self.sum_pixels[ri, :] / self.count[ri] - self.sum_pixels[rj, :] / self.count[rj]
            diff = np.sqrt(np.sum(d * d))
            if diff < self.thr:
                self.S[rj] = ri
                self.count[ri] += self.count[rj]
                self.count[rj] = 0
                self.sum_pixels[ri, :] += self.sum_pixels[rj, :]
                self.sum_pixels[rj, :] = 0

    def regular_union(self, i, j):
        # Joins two regions if they are similar
        # Keeps track of size and mean color of regions
        ri = self.find(i)
        rj = self.find(j)
        if ri != rj:
            self.S[rj] = ri
            self.count[ri] += self.count[rj]
            self.count[rj] = 0
            self.sum_pixels[ri, :] += self.sum_pixels[rj, :]
            self.sum_pixels[rj, :] = 0

    def connected_components_segmentation(self):
        # rows = I.shape[0]
        # cols = I.shape[1]
        for p in range(self.S.shape[0]):
            if p % self.cols < self.cols - 1:  # If p is not in the last column
                self.union(p, p + 1)  # p+1 is the pixel to the right of p
            if p // self.cols < self.rows - 1:  # If p is not in the last row
                self.union(p, p + self.cols)  # p+cols is the pixel to below p

    def calculate_threshold(self):
        std = [np.std(self.I[:, :, 0]), np.std(self.I[:, :, 1]), np.std(self.I[:, :, 2])]
        self.thr = std[np.argmax(std)]
        # return std[np.argmax(std)]

    def num_regions(self):
        return np.sum(self.S == -1)

    def regions(self, size=1):
        """
        Returns regions of size n in image
        """
        return np.sum(self.count == size)

    def show(self, mode='mean'):
        """
        Show image using cv2.imshow, mode is either mean of regions or
        random colors
        """
        rand_cm = np.random.random_sample((self.rows * self.cols, 3))
        seg_im_mean = np.zeros((self.rows, self.cols, 3))
        seg_im_rand = np.zeros((self.rows, self.cols, 3))
        for r in range(self.rows - 1):
            for c in range(self.cols - 1):
                f = self.find(r * self.cols + c)
                seg_im_mean[r, c, :] = self.sum_pixels[f, :] / self.count[f]
                seg_im_rand[r, c, :] = rand_cm[f, :]

        if mode == 'mean':
            cv2.imshow('Segmentation - using mean colors', seg_im_mean)
        elif mode == 'rand':
            cv2.imshow('Segmentation - using random colors', seg_im_rand)
