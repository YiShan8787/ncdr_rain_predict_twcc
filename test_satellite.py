#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 14:42:03 2021

@author: ubuntu
"""

import cv2

satellite_path = "/media/ubuntu/My Passport/NCDR/ncdr_rain_predict/data/satellite/2015/07/s1q201507010000.jpg"

img = cv2.imread(satellite_path)  
img = img[130:340,:-60]
cv2.imshow("test",img)
 
cv2.waitKey()
cv2.destroyAllWindows()