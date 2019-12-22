import numpy as np
import cv2
import matplotlib.pyplot as plt
#%matplotlib inline
from scipy import ndimage


def remove_isolated_pixels(image):
    connectivity = 8

    output = cv2.connectedComponentsWithStats(image, connectivity, cv2.CV_32S)

    num_stats = output[0]
    labels = output[1]
    stats = output[2]

    new_image = image.copy()

    for label in range(num_stats):
        if stats[label,cv2.CC_STAT_AREA] == 1:
            new_image[labels == label] = 0

    return new_image

img = cv2.imread("./multi_plant/rgb_00_00_006_00.png")
# plt.imshow(img)
# plt.show()

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# # Green color
# Image filtered in HSV between [25 39 43] and [ 62 255 122].
low_green = np.array([25, 39, 43])
high_green = np.array([62, 255, 122])

green_mask = cv2.inRange(hsv, low_green, high_green)
green = cv2.bitwise_and(img, img, mask=green_mask)
# plt.imshow(green)
# # plt.show()
binary = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(binary, (5, 5), 0)
ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

plt.imshow(th3, cmap='gray', vmin=0, vmax=255)
plt.show()
#
#
#
# # plt.subplot(1, 2, 1)
# # plt.imshow(mask, cmap="gray")
# # plt.subplot(1, 2, 2)
# grey = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)
# plt.imshow(grey, cmap = plt.cm.gray)
# plt.show()
