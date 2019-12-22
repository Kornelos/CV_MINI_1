import numpy as np
import cv2
import matplotlib.pyplot as plt
from colorfilters import HSVFilter
img = cv2.imread("./multi_plant/rgb_00_00_006_00.png")
window = HSVFilter(img)
window.show()
print(f"Image filtered in HSV between {window.lowerb} and {window.upperb}.")
# while(True):
#
#     # grab the frame
#     #ret, frame = cap.read()
#
#     # get trackbar positions
#     ilowH = cv2.getTrackbarPos('lowH', 'image')
#     ihighH = cv2.getTrackbarPos('highH', 'image')
#     ilowS = cv2.getTrackbarPos('lowS', 'image')
#     ihighS = cv2.getTrackbarPos('highS', 'image')
#     ilowV = cv2.getTrackbarPos('lowV', 'image')
#     ihighV = cv2.getTrackbarPos('highV', 'image')
#
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     lower_hsv = np.array([ilowH, ilowS, ilowV])
#     higher_hsv = np.array([ihighH, ihighS, ihighV])
#     mask = cv2.inRange(hsv, lower_hsv, higher_hsv)qq
#
#     frame = cv2.bitwise_and(frame, frame, mask=mask)
#
#     # show thresholded image
#     cv2.imshow('image', frame)
#     k = cv2.waitKey(1000) & 0xFF # large wait time to remove freezing
#     if k == 113 or k == 27:
#         break
