import numpy as np
import cv2
import matplotlib.pyplot as plt


# i dont remember writing this, maybe works
def remove_isolated_pixels(image):
    connectivity = 8

    output = cv2.connectedComponentsWithStats(image, connectivity, cv2.CV_32S)

    num_stats = output[0]
    labels = output[1]
    stats = output[2]

    new_image = image.copy()

    for label in range(num_stats):
        if stats[label, cv2.CC_STAT_AREA] == 1:
            new_image[labels == label] = 0

    return new_image




img = cv2.imread("./multi_plant/rgb_00_00_006_00.png")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# # Green color
# Image filtered in HSV between [25 39 43] and [ 62 255 122].
# 2nd filter Image filtered in HSV between [28 39 56] and [ 62 255 137].
# Image filtered in HSV between [25 38 42] and [ 72 255 255].
low_green = np.array([25, 38, 42])
high_green = np.array([72, 255, 255])

green_mask = cv2.inRange(hsv, low_green, high_green)
green = cv2.bitwise_and(img, img, mask=green_mask)



# smooth img
blur = cv2.GaussianBlur(green_mask, (5, 5), 0)
# ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

kernel = np.ones((5,5), np.uint8)
# erosion - deletes noise dilitation - THICC
img_erosion = cv2.erode(green_mask, kernel, iterations=1)
img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)

plt.imshow(img_erosion)
plt.show()
plt.imshow(img_dilation)
plt.show()

# extract contours
contours, hierarchy = cv2.findContours(img_dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, 0, (0, 0, 255), 3)

cont_max = max(contours, key=cv2.contourArea)
cv2.drawContours(img, cont_max, -1, (0, 255, 0), 3)
plt.imshow(img)
plt.show()
print(img.shape)
empty = np.zeros((480,480))
# draw filled plant
cv2.fillPoly(empty, pts =[cont_max], color=(255,255,255))
plt.imshow(empty)
plt.show()
