import numpy as np
import cv2
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

# global switch for development mode
DEBUG = False


def process_img(dir, filename):
    img = cv2.imread(dir + filename)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # # Green color
    low_green = np.array([35, 44, 25])
    high_green = np.array([73, 255, 255])
    green_mask = cv2.inRange(hsv, low_green, high_green)
    if DEBUG:
        plt.title('green_mask')
        plt.imshow(green_mask)
        plt.show()

    # extract contours
    contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, 0, (0, 0, 255), 3)
    cont_max = max(contours, key=cv2.contourArea)
    # cv2.drawContours(img, cont_max, -1, (0, 255, 0), 3)
    if DEBUG:
        plt.imshow(img)
        plt.show()
    empty = np.zeros(green_mask.shape)
    # draw filled plant
    cv2.fillPoly(empty, pts=[cont_max], color=(255, 255, 255))
    empty = empty.astype('uint8')
    if DEBUG:
        plt.title('fill_poly (contures)')
        plt.imshow(empty)
        plt.show()
    #bounding box
    if DEBUG:
        rect = cv2.minAreaRect(cont_max)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img, [box], -1, (0, 255, 0), 3)
        plt.imshow(img)
        plt.show()

    kernel = np.ones((5, 5), np.uint8)
    # erosion - deletes noise, dilitation - thickens (not used)
    img_erosion = cv2.erode(empty, kernel, iterations=1)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)

    if DEBUG:
        plt.title('erosion')
        plt.imshow(img_erosion)
        plt.show()
        plt.title('dilitation')
        plt.imshow(img_dilation)
        plt.show()



    # # WATERSHED ALGORITHM
    # # noise removal
    # empty = empty.astype('uint8')
    # kernel = np.ones((3, 3), np.uint8)
    # opening = cv2.morphologyEx(empty, cv2.MORPH_OPEN, kernel, iterations=2)
    # # sure background area
    # sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # # Finding sure foreground area
    # dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # # Finding unknown region
    # sure_fg = np.uint8(sure_fg)
    # unknown = cv2.subtract(sure_bg, sure_fg)
    # # Marker labelling
    # ret, markers = cv2.connectedComponents(sure_fg)
    # # Add one to all labels so that sure background is not 0, but 1
    # markers = markers + 1
    # # Now, mark the region of unknown with zero
    # markers[unknown == 255] = 0
    # markers = cv2.watershed(img, markers)
    # # img[markers == -1] = [255, 0, 0]
    # if DEBUG:
    #     plt.imshow(markers)
    #     plt.show()
    # cv2.imwrite(filename, empty,)
    # remove contents
    # list(map(unlink, (join('./output/', f) for f in listdir('./output/'))))
    # write results
    cv2.imwrite(join('./output/', filename), empty)


mypath = './multi_plant/'
images = [f for f in listdir(mypath) if isfile(join(mypath, f))]
if not DEBUG:
    for img in images:
        process_img(mypath, img)

if DEBUG:
    process_img(mypath, images[30])
