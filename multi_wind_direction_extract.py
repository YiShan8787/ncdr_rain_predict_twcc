# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 20:04:05 2021

@author: user
"""
import os
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy import interpolate
import cv2

station_path = '/home/om990301/ncdr_rain_predict/data/station_data'

max_wind_direction = 360
min_wind_direction = 0

black_list_station = ['C0S730','467620']

def mkdir(create_path):
    #判斷目錄是否存在
    #存在：True
    #不存在：False
    folder = os.path.exists(create_path)

    #判斷結果
    if not folder:
        #如果不存在，則建立新目錄
        os.makedirs(create_path)
        print('-----建立成功-----')

    else:
        #如果目錄已存在，則不建立，提示目錄已存在
        #print(create_path+'目錄已存在')
        pass

for year in os.listdir(station_path):
    #print(file)
    year_dir = station_path + "/" + year
    for month in os.listdir(year_dir):
        month_dir = year_dir + "/" + month
        for date in os.listdir(month_dir):
            date_dir = month_dir + "/" + date
            for date_file in os.listdir(date_dir):
                if not date_file.endswith(".txt"):
                    #break
                    continue
                file_name = date_file
                date_txt = date_dir + "/" + date_file
                print(date_txt)
                f = open(date_txt)
                
                stations = []
                lons = []
                lats = []
                elevs = []
                temps = []
                wind_directions = []
                
                for line in f.readlines():
                    line = line.replace("-", " -")
                    parsing = line.split()
                    if parsing[0] in black_list_station:
                        continue
                    # parsing append list
                    stations.append(parsing[0])
                    lons.append(float("{:.2f}".format(float(parsing[1]))))
                    lats.append(float("{:.2f}".format(float(parsing[2]))))
                    #elevs.append(float("{:.4f}".format(float(parsing[3]))))
                    elevs.append(float(parsing[3]))
            
                    temps.append(float(parsing[5]))
            
                    wind_directions.append((float(parsing[7])))
                    
            
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
                wind_direction_2d = np.full((math.ceil(lons_diff/resolution), math.ceil(lats_diff/resolution)), np.nan)
                
                #temps_2d = np.full((math.ceil(lats_diff/resolution),math.ceil(lons_diff/resolution)), np.nan)
                #wind_direction_2d = np.full((math.ceil(lats_diff/resolution),math.ceil(lons_diff/resolution)), np.nan)
                
                for i in range(len(lons)):
                    tmp_lon_idx = math.floor((lons[i] - lons_min)/resolution)
                    tmp_lat_idx = math.floor((lats[i] - lats_min)/resolution)
                    #print("idx_x,idx_y: ",tmp_lon_idx,tmp_lat_idx)
                    wind_direction_2d[tmp_lon_idx,tmp_lat_idx] = wind_directions[i]
                    #wind_direction_2d[tmp_lat_idx,tmp_lon_idx] = wind_directions[i]
                    
                
                #plt.imshow(wind_direction_2d,interpolation='nearest')
                #plt.show()
                
                wind_direction_2d_delxy = wind_direction_2d[169:,:340]
                
                #plt.imshow(wind_direction_2d_delxy,interpolation='nearest')
                #plt.show()
                
                wind_direction_2d_delxy[wind_direction_2d_delxy< -20] = np.nan
                
                x = np.arange(0, wind_direction_2d_delxy.shape[1])
                y = np.arange(0, wind_direction_2d_delxy.shape[0])
                #mask invalid values
                mask_array = np.ma.masked_invalid(wind_direction_2d_delxy)
                xx, yy = np.meshgrid(x, y)
                #get only the valid values
                x1 = xx[~mask_array.mask]
                y1 = yy[~mask_array.mask]
                newarr = mask_array[~mask_array.mask]
                
                inter_wind_direction_arr = interpolate.griddata((x1, y1), newarr.ravel(),
                                          (xx, yy),
                                             method='cubic')
                
                inter_wind_direction_arr[inter_wind_direction_arr>max_wind_direction] = max_wind_direction
                inter_wind_direction_arr[inter_wind_direction_arr<min_wind_direction] = min_wind_direction
                inter_wind_direction_arr = inter_wind_direction_arr/max_wind_direction
                
                mkdir(date_dir + "/wind_direction_img")
                mkdir(date_dir + "/wind_direction_npy")
                
                #plt.imshow(inter_wind_direction_arr,interpolation='nearest')
                #plt.colorbar()
                #plt.savefig(date_dir +"/wind_direction_img/" +  date_file.split(".")[0] +  ".png")
                #plt.show()

                #inter_wind_direction_arr[inter_wind_direction_arr == np.nan] = 0
                np.nan_to_num(inter_wind_direction_arr, copy=False)
                gray_three_channel = cv2.cvtColor(inter_wind_direction_arr.astype('float32'),cv2.COLOR_GRAY2RGB)
                
                np.save(date_dir +"/wind_direction_npy/" + date_file.split(".")[0] + "_wind_direction_arr",gray_three_channel)