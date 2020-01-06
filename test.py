import numpy as np
import cv2
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from colors import colors
from random import randint

# global switch for development mode
DEBUG = False


def process_img(f_dir, filename):
    img = cv2.imread(f_dir + filename)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # HSV based threshold
    low_green = np.array([35, 44, 25])
    high_green = np.array([73, 255, 255])
    green_mask = cv2.inRange(hsv, low_green, high_green)
    if DEBUG:
        plt.title('green_mask')
        plt.imshow(green_mask)
        plt.show()

    # extract contours
    contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # find longest
    cont_max = max(contours, key=cv2.contourArea)
    empty = np.zeros(green_mask.shape)
    # draw filled plant
    cv2.fillPoly(empty, pts=[cont_max], color=(255, 255, 255))
    empty = empty.astype('uint8')
    if DEBUG:
        plt.title('fill_poly (contures)')
        plt.imshow(empty)
        plt.show()
    # bounding box
    if DEBUG:
        img_box = img.copy()
        rect = cv2.minAreaRect(cont_max)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img_box, [box], -1, (0, 255, 0), 3)
        plt.imshow(img_box)
        plt.show()

    # WATERSHED
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(empty, cv2.MORPH_OPEN, kernel, iterations=1)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=5)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.25 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)

    # drawing after watershed
    img[markers < 2] = [0, 0, 0]
    if np.unique(markers).__len__() - 2 <= 7:
        for i in range(0, 7):
            img[markers == i + 2] = colors[i]
    else:
        for i in range(0, np.unique(markers).__len__() - 2):
            img[markers == i + 2] = [
                randint(0, 255), randint(0, 255), randint(0, 255)]
    if DEBUG:
        plt.imshow(img)
        plt.show()
    # write results

    # threshold
    cv2.imwrite(join('./output/', filename), empty)
    # segmentation
    cv2.imwrite(join('./output_seg/', filename), img)


file_path = './multi_plant/'
images = [f for f in listdir(file_path) if isfile(join(file_path, f))]
if not DEBUG:
    for im in images:
        process_img(file_path, im)

if DEBUG:
    process_img(file_path, images[357])
