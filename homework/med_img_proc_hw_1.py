import sys
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from utills import *

head = cv2.imread('imgs/isch_head.bmp',cv2.IMREAD_GRAYSCALE)
seed = (180, 95)

(seg, size) = region_growing(head, seed, 20)
print(size)
plt.figure()
plt.subplot(1,2,1)
plt.imshow(head, cmap='gray')
plt.title('Original Image')
plt.subplot(1,2,2)
plt.imshow(seg, cmap='gray')
plt.title('After region growing segmentation')
print(plt.ginput())
plt.show()
