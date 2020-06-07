import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import measure,morphology

# open files:
img = cv2.imread('imgs/Liver_CT.bmp',cv2.IMREAD_GRAYSCALE)
binary = cv2.imread('imgs/Binary.bmp',cv2.IMREAD_GRAYSCALE)

# open morphological operator and connected components algorithm:
binary_cleaned = morphology.opening(binary, morphology.disk(15))
bin_lbls = measure.label(binary_cleaned)
regions = measure.regionprops(bin_lbls)

# search for liver in labeled regions:
max_area = 0
liver_item = None
for item in regions:
    if max_area<item.area:
        max_area = item.area
        liver_item = item
num_of_pixels_in_liver = liver_item.area

# create liver only image:
liver_only = np.zeros(binary.shape)
liver_only[bin_lbls==liver_item.label] = 255

# add counter of liver to original image:
contour = liver_only - morphology.erosion(liver_only, morphology.disk(1))
img_with_contour = img+contour

plt.figure()
plt.subplot(2,1,1)
plt.imshow(liver_only, cmap='gray')
plt.title('Liver - '+str(num_of_pixels_in_liver)+' pixels')
plt.subplot(2,1,2)
plt.imshow(img_with_contour, cmap='gray')
plt.title('Original image with contour')
plt.show()

# plt.figure()
# plt.subplot(2,3,1)
# plt.imshow(binary, cmap='gray')
# plt.title('Original Image')
# plt.subplot(2,3,2)
# plt.imshow(binary_cleaned, cmap='gray')
# plt.title('After erosion')
# plt.subplot(2,3,3)
# plt.imshow(bin_lbls, cmap='gray')
# plt.title('After labels')
# plt.subplot(2,3,4)
# plt.imshow(liver_only, cmap='gray')
# plt.title('Liver only')
# plt.subplot(2,3,5)
# plt.imshow(countor, cmap='gray')
# plt.title('Countor')