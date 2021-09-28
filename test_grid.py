# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 12:45:35 2021

@author: user
"""
import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt

#Let's create some random  data
array = np.random.random_integers(0,10,(10,10)).astype(float)
#values grater then 7 goes to np.nan
array[array>7] = np.nan

plt.imshow(array,interpolation='nearest')
plt.show()

x = np.arange(0, array.shape[1])
y = np.arange(0, array.shape[0])
#mask invalid values
mask_array = np.ma.masked_invalid(array)
xx, yy = np.meshgrid(x, y)
#get only the valid values
x1 = xx[~mask_array.mask]
y1 = yy[~mask_array.mask]
newarr = array[~mask_array.mask]
print(type(newarr))

GD1 = interpolate.griddata((x1, y1), newarr.ravel(),
                          (xx, yy),
                             method='cubic')

plt.imshow(GD1,interpolation='nearest')
plt.show()