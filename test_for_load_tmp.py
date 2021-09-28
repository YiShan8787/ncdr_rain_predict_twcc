# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 15:24:10 2021

@author: user
"""

import os
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy import interpolate

station_path = "E:\\tech\\ncdr\\ncdr_rain_predict\\data\\station_data\\2016\\05\\20160501"

for file in os.listdir(station_path):
    #print(file)
    file_name = station_path + "\\" + file
    f = open(file_name)
    stations = []
    lons = []
    lats = []
    elevs = []
    temps = []
    huminitys = []
    
    for line in f.readlines():
        line = line.replace("-", " -")
        parsing = line.split()
        
        # parsing append list
        stations.append(parsing[0])
        lons.append(float("{:.2f}".format(float(parsing[1]))))
        lats.append(float("{:.2f}".format(float(parsing[2]))))
        #elevs.append(float("{:.4f}".format(float(parsing[3]))))
        elevs.append(float(parsing[3]))

        temps.append(float(parsing[5]))

        huminitys.append((float(parsing[6])))
        

lons_min = min(lons)
lons_max = max(lons)
lons_diff = lons_max - lons_min

lats_min = min(lats)
lats_max = max(lats)
lats_diff = lats_max - lats_min

resolution = 0.01


#check duplicate
location_list = []
for i in range(len(lons)):
    location = str(lons[i]) + "," + str(lats[i])
    if location not in location_list:
        location_list.append(location)
    #else:
        #print("duplicate location, old_idx, old_station, new_idx, new_station: (",location + ")", location_list.index(location),stations[location_list.index(location)],i,stations[i])
        



temps_2d = np.full((math.ceil(lons_diff/resolution), math.ceil(lats_diff/resolution)), np.nan)
huminity_2d = np.full((math.ceil(lons_diff/resolution), math.ceil(lats_diff/resolution)), np.nan)

#temps_2d = np.full((math.ceil(lats_diff/resolution),math.ceil(lons_diff/resolution)), np.nan)
#huminity_2d = np.full((math.ceil(lats_diff/resolution),math.ceil(lons_diff/resolution)), np.nan)

for i in range(len(lons)):
    tmp_lon_idx = math.floor((lons[i] - lons_min)/resolution)
    tmp_lat_idx = math.floor((lats[i] - lats_min)/resolution)
    #print("idx_x,idx_y: ",tmp_lon_idx,tmp_lat_idx)
    huminity_2d[tmp_lon_idx,tmp_lat_idx] = huminitys[i]
    #huminity_2d[tmp_lat_idx,tmp_lon_idx] = huminitys[i]
    

plt.imshow(huminity_2d,interpolation='nearest')
plt.show()

huminity_2d_delxy = huminity_2d[169:,:340]

plt.imshow(huminity_2d_delxy,interpolation='nearest')
plt.show()

huminity_2d_delxy[huminity_2d_delxy< -20] = np.nan

x = np.arange(0, huminity_2d_delxy.shape[1])
y = np.arange(0, huminity_2d_delxy.shape[0])
#mask invalid values
mask_array = np.ma.masked_invalid(huminity_2d_delxy)
xx, yy = np.meshgrid(x, y)
#get only the valid values
x1 = xx[~mask_array.mask]
y1 = yy[~mask_array.mask]
newarr = mask_array[~mask_array.mask]

inter_huminity_arr = interpolate.griddata((x1, y1), newarr.ravel(),
                          (xx, yy),
                             method='cubic')

plt.imshow(inter_huminity_arr,interpolation='nearest')
plt.colorbar()
plt.show()