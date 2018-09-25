# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as matio
import os
import cv2
import torch
import matplotlib.patches as mpatches
from skimage import io


def load_mat(path):
    with open(path) as f:
        mat = matio.loadmat(f)
    return mat

def create_patch(pos):
    xy = (pos[0], pos[1])
    w = pos[2] - pos[0]
    h = pos[3] - pos[1]
    return mpatches.Rectangle(xy, w, h, alpha=0.7)

def create_circle(xy):
    x = xy[0]
    y = xy[1]
    return mpatches.Circle((x,y), radius=5)

def rescale_bbox(bbox):
    midx = (bbox[0] + bbox[2]) / 2.0
    midy = (bbox[1] + bbox[3]) / 2.0
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    s = 1.25
    w *= s
    h *= s
    return [midx - w / 2.0, midy - h / 2.0, 
            midx + w / 2.0, midy + h / 2.0]

def revert_img(img):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    img = img * std + mean
    img *= 255
    img = np.clip(img, 0, 255)
    return img

def get_max(a1, a2):
    return max(np.max(a1), np.max(a2))

def get_min(a1, a2):
    return min(np.min(a1), np.min(a2))

fig, ax1 = plt.subplots(figsize=(6,6))
ax1.imshow(heatmap)
p1 = create_patch(p_bbox)
p2 = create_patch(p_corner)
ax1.add_patch(p1)
ax1.add_patch(p2)

fig, ax2 = plt.subplots(figsize=(6,6))
ax2.imshow(heatmap)
c1 = create_circle([p1x, p1y])
c2 = create_circle([p2x, p2y])
ax2.add_patch(c1)
ax2.add_patch(c2)
