# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 12:32:00 2021

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 12:45:35 2021

@author: user
"""

import numpy as np
import argparse
import cv2

from imutils import paths
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--weather_dataset", required=False,
	help="path to input dataset", default = "E:\\tech\\ncdr\\ncdr_rain_predict\\cut_satellite")

args = vars(ap.parse_args())

print("[INFO] loading weather images...")
imagePaths = list(paths.list_images(args["weather_dataset"]))

# 裁切區域的 x 與 y 座標（左上角）
x = 0
y = 130

y_len = 340
x_len = -60


# loop over the image paths
for imagePath in imagePaths:
    # extract the class label from the filename
    print(imagePath)
    label = imagePath.split(os.path.sep)[-1][3:11]
    # load the image, swap color channels, and resize it to be a fixed
    # 224x224 pixels while ignoring aspect ratio
    image = cv2.imread(imagePath)
    #if isinstance(image,type(None)):
    #    print("error")
    #print(type(image))
    #image = cv2.resize(image, (224, 224))
    image = image[y:340,x:-60]
    #print(image.shape)
    # update the data and labels lists, respectively
    #if time == 0:
    cv2.imshow(label, image)
    cv2.waitKey()
    
cv2.destroyAllWindows()