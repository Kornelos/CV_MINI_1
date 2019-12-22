import cv2
from colorfilters import HSVFilter
img = cv2.imread("./multi_plant/rgb_00_00_006_00.png")
window = HSVFilter(img)
window.show()
print(f"Image filtered in HSV between {window.lowerb} and {window.upperb}.")
