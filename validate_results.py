import cv2
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join


def verify(res, grand_t):
    img = cv2.imread(grand_t)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, gt) = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)

    img = cv2.imread(res)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, seg) = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)

    k = 255

    dice = np.sum(seg[gt==k])*2.0 / (np.sum(seg[seg==k]) + np.sum(gt[gt==k]))
    # print('Dice similarity score is {}'.format(dice))
    return dice


res_path = './output/'
res_images = [f for f in listdir(res_path) if isfile(join(res_path, f))]

label_path = './multi_label/'
label_images = [f for f in listdir(label_path) if isfile(join(label_path, f))]

dices = []

for i in range(0, 900):
    dices.append(verify(res_path + res_images[i], label_path + label_images[i]))

print('mean dice similarity score {}'.format(np.mean(dices)))

# plant = 0
# for i in range(1, 15):
#     print('mean per plant ' + plant.__str__() + ': {}'.format(np.mean(dices[60*(i-1):60*i-1])))
#     plant += 1
#     plant = plant % 5


# 0 0.917151
# 73 0.9175702797989728