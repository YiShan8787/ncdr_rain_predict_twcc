# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 11:27:15 2021

@author: user
"""

# import the necessary packages

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import concatenate

from tensorflow.keras.layers import MaxPooling1D, Conv1D

from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed



from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

from openpyxl import load_workbook
import time as tt

from tensorflow import keras

#os.environ["CUDA_VISIBLE_DEVICES"] = "" # use cpu
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
###############################################################################
#----------------------------PATH---------------------------
# weather image path
ap.add_argument("-wd", "--weather_dataset", required=False,
	help="path to input dataset", default = "/home/om990301/ncdr_rain_predict/data/weather_image")
ap.add_argument("-sd", "--satellite_dataset", required=False,
	help="path to input dataset", default = "/home/om990301/ncdr_rain_predict/data/satellite")
# ground truth path
ap.add_argument("-gt", "--gt", type=str, default="/home/om990301/ncdr_rain_predict/data/gt/south.xlsx",
	help="gt for the data")
station_path = '/home/om990301/ncdr_rain_predict/data/station_data'
model_path = "/home/om990301/ncdr_rain_predict/Model/rain_predict.h5"
result_path = "Result/result.txt"
#--------------------------data parameter----------------------
# station data time
ap.add_argument("-st", "--station_time", type=int, default=12,
	help="time for station data")


use_sampling = True

satellite_frame = -10
#north '466900','466940','466920'
#mid 'C0G860','C01460'??
#south 'C0V250','01O760'??,'C0R140'
special_station_input_id = ['C0V250','C0R140']

#################################################################################

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-3
EPOCHS = 10
BS = 1
num_folds = 3

random_st = 42
# loss and acc file name
ap.add_argument("-p", "--plot", type=str, default="loss_acc.png",
	help="path to output loss/accuracy plot")
# model file name
ap.add_argument("-m", "--model", type=str, default="rain_predict.h5",
	help="path to output loss/accuracy plot")

satellite_x = 210
satellite_y = 340

args = vars(ap.parse_args())

print("[INFO] loading gt")
gtPaths = args["gt"]
station_time = args["station_time"]

# 讀取 Excel 檔案
wb = load_workbook(gtPaths)
sheet = wb.active
check_file_cnt = 1

gt_time_list = [] 

for row in sheet.rows:
    #print(row[0].value)
    if str(row[0].value) not in gt_time_list:
        gt_time_list.append(str(row[0].value))
    #for cell in row:
    #    print(cell.value)

del wb
del sheet
# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading weather images...")
imagePaths = list(paths.list_images(args["weather_dataset"]))
data = []
labels = []

last_label = None
data_weather_labels = []
data_weathers = []
tmp_weathers = []

# 裁切區域的 x 與 y 座標（左上角）
x = 220
y = 163

y_len = 340
x_len = 210

# loop over the image paths
for imagePath in imagePaths:
    # extract the class label from the filename
    #print(imagePath)
    label = imagePath.split(os.path.sep)[-1]
    #print(label[-8:-4])
    time = label[-8:-4]
    if time == "1200" or time == "1800" or time =="0600":
        continue
    
    image = cv2.imread(imagePath)
    
    image = image[y:y+y_len,x:x+x_len]
    
    cv2.waitKey()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    data.append(image)
    labels.append(label)
    
    if label[3:11] not in data_weather_labels:
        data_weather_labels.append(label[3:11])
        #print(imagePath.split(os.path.sep))
    
    

#print(len(labels))
# convert the data and labels to NumPy arrays while scaling the pixel
# intensities to the range [0, 1]
#print(len(data))
data_weathers = np.reshape(data, (-1,1,340,210,3))
data = np.array(data) / 255.0
#labels = np.array(labels)
cv2.destroyAllWindows()
data_weathers = np.array(data_weathers) / 255.0
#data_weather_labels = np.array(data_weather_labels)
print("number of videos: ", str(len(data_weathers)))
print("number of the date of the videos: ", str(data_weathers.shape[0]))
print(data_weathers.shape)

#del labels
del data



print("[INFO] loading station data(huminity)")
#loading station huminity

tmp_huminitys = []


for year in os.listdir(station_path):
    #print(file)
    year_dir = station_path + "/" + year
    for month in os.listdir(year_dir):
        month_dir = year_dir + "/" + month
        for date in sorted(os.listdir(month_dir)):
            date_cnt = int(date[-2:])
            #print(date_cnt)
            
            if date not in data_weather_labels:
                data_weather_labels.append(date)
            
            if date_cnt == 1:
                check_file_cnt = date_cnt
            else:
                while check_file_cnt+1 != date_cnt:
                    #print(date_cnt)
                    #print(check_file_cnt)
                    for file_i in range(station_time):
                        tmp_huminitys.append(np.zeros((210,340,3)))
                    check_file_cnt = check_file_cnt +1
                    print("lack file before: ",date)
                check_file_cnt = date_cnt
            
            date_dir = month_dir + "/" + date + "/huminity_npy"
            for date_file in os.listdir(date_dir):
                if not date_file.endswith(".npy"):
                    continue
                #print(date_file[8:10])
                time = int(date_file[8:10])
                if time >11:
                    continue
                    #print(time)
                file_name = date_file
                date_txt = date_dir + "/" + date_file
                #print(date_txt)
                f = np.load(date_txt)
                #f[f == np.nan] = 0
                
                tmp_huminitys.append(f)
            
data_station_huminity = np.array(tmp_huminitys)
data_station_huminity = np.reshape(data_station_huminity,(-1,station_time,210,340,3))
del tmp_huminitys
print("number of videos: ", data_station_huminity.shape[0])


#find time in labels, then buid the frames
# loading groundtruth
print("[INFO] category labeling")
category_labels = []
positive_number = 0
sample_list = []
for i in range(len(data_weather_labels)):
    if data_weather_labels[i] in gt_time_list:
        category_labels.append(1)
        positive_number = positive_number +1
    else:
        category_labels.append(0)
        sample_list.append(i)
    
    
print("number of positive number: ", str(positive_number))
print("number of negative number: ", str(len(data_weather_labels) - positive_number))
# perform one-hot encoding on the labels
print("[INFO] one-hot")
category_labels = to_categorical(category_labels)


print("[INFO] loading station data(temperature)")
# loading station temp
#data_station_huminity =[]
tmp_temps = []


for year in os.listdir(station_path):
    #print(file)
    year_dir = station_path + "/" + year
    for month in os.listdir(year_dir):
        month_dir = year_dir + "/" + month
        for date in sorted(os.listdir(month_dir)):
            date_dir = month_dir + "/" + date + "/temp_npy"

            date_cnt = int(date[-2:])
            #print(date_cnt)
            
            if date_cnt == 1:
                check_file_cnt = date_cnt
            else:
                while check_file_cnt+1 != date_cnt:
                    #print(date_cnt)
                    #print(check_file_cnt)
                    for file_i in range(station_time):
                        tmp_temps.append(np.zeros((210,340,3)))
                    check_file_cnt = check_file_cnt +1
                    print("lack file before: ",date)
                check_file_cnt = date_cnt

            for date_file in os.listdir(date_dir):
                if not date_file.endswith(".npy"):
                    continue
                time = int(date_file[8:10])
                if time >11:
                    continue
                    #print(time)
                file_name = date_file
                date_txt = date_dir + "/" + date_file
                #print(date_txt)
                f = np.load(date_txt)
                #f[f == np.nan] = 0
                
                
                tmp_temps.append(f)
            
data_station_temperature = np.array(tmp_temps)
data_station_temperature = np.reshape(data_station_temperature,(-1,station_time,210,340,3))
print("number of videos: ", data_station_temperature.shape[0])
del tmp_temps

print("[INFO] loading station data(wind_direction)")
#loading station wind direction
tmp_wind_directions = []


for year in os.listdir(station_path):
    #print(file)
    year_dir = station_path + "/" + year
    for month in sorted(os.listdir(year_dir)):
        month_dir = year_dir + "/" + month
        for date in sorted(os.listdir(month_dir)):
            date_dir = month_dir + "/" + date + "/wind_direction_npy"

            date_cnt = int(date[-2:])
            #print(date_cnt)
            
            if date_cnt == 1:
                check_file_cnt = date_cnt
            else:
                while check_file_cnt+1 != date_cnt:
                    #print(date_cnt)
                    #print(check_file_cnt)
                    for file_i in range(station_time):
                        tmp_wind_directions.append(np.zeros((210,340,3)))
                    check_file_cnt = check_file_cnt +1
                    print("lack file before: ",date)
                check_file_cnt = date_cnt

            for date_file in os.listdir(date_dir):
                if not date_file.endswith(".npy"):
                    continue
                time = int(date_file[8:10])
                if time >11:
                    continue
                    #print(time)
                file_name = date_file
                date_txt = date_dir + "/" + date_file
                #print(date_txt)
                f = np.load(date_txt)
                #f[f == np.nan] = 0
                
                
                tmp_wind_directions.append(f)
            
data_station_wind_direction = np.array(tmp_wind_directions)
data_station_wind_direction = np.reshape(data_station_wind_direction,(-1,station_time,210,340,3))
print("number of videos: ", data_station_wind_direction.shape[0])
del tmp_wind_directions

print("[INFO] loading satellite data")
# loading satellite data
tmp_satellite = []
data_satellite = []
#cnt = 0

satellite_path = args["satellite_dataset"]

for year in os.listdir(satellite_path):
    year_dir = satellite_path + "/" + year
    for month in os.listdir(year_dir):
        month_dir = year_dir + "/" + month
        #cnt = 0
        #print(month)
        for date in sorted(os.listdir(month_dir)):
            date_dir = month_dir + "/" + date 
            #time = date[-8:-4]
            #print(time)
            time = int(date[-8:-4])
            if time >1200:
                if tmp_satellite:
                    #print(len(tmp_satellite))
                    #print(date_dir)
                    data_satellite.append(tmp_satellite[satellite_frame:])
                    #print(np.array(data_satellite).shape)
                    tmp_satellite = []
                    #cnt = cnt +1
                    #print("update satellite")
                continue
                
            
            f = cv2.imread(date_dir)
            f = f[130:340,:-60]
            tmp_satellite.append(f)
        #print(cnt)
            #data_station_huminity = np.reshape()
                #print(f.shape)
data_satellite = np.array(data_satellite)
print("number of videos: ", data_satellite.shape)
del tmp_satellite

print("[INFO] loading special station")
#loading single station data
tmp_special_stations = []
data_special_stations = []

for year in os.listdir(station_path):
    #print(file)
    year_dir = station_path + "/" + year
    for month in os.listdir(year_dir):
        month_dir = year_dir + "/" + month
        for date in sorted(os.listdir(month_dir)):
            date_dir = month_dir + "/" + date

            date_cnt = int(date[-2:])
            #print(date_cnt)
            
            if date_cnt == 1:
                check_file_cnt = date_cnt
            else:
                while check_file_cnt+1 != date_cnt:
                    #print(date_cnt)
                    #print(check_file_cnt)
                    for file_i in range(station_time):
                        for num in range(len(stations)):
                            tmp_special_stations.append([0,0,0])
                        data_special_stations.append(tmp_special_stations)
                        tmp_special_stations = []
                    check_file_cnt = check_file_cnt +1
                    
                    print("lack file before: ",date)
                check_file_cnt = date_cnt

            for date_file in os.listdir(date_dir):
                if not date_file.endswith(".txt"):
                    continue
                time = int(date_file[-6:-4])
                if time >11:
                    continue
                file_name = date_file
                date_txt = date_dir + "/" + date_file
                #print(date_txt)
                f = open(date_txt)
                
                stations = []
                lons = []
                lats = []
                elevs = []
                temps = []
                huminitys = []
                wind_directions = []
                
                for line in f.readlines():
                    line = line.replace("-", " -")
                    parsing = line.split()
                    if parsing[0] not in special_station_input_id:
                        continue
                    else:
                        # parsing append list
                        stations.append(parsing[0])
                        lons.append(float("{:.2f}".format(float(parsing[1]))))
                        lats.append(float("{:.2f}".format(float(parsing[2]))))
                        #elevs.append(float("{:.4f}".format(float(parsing[3]))))
                        elevs.append(float(parsing[3]))
                
                        temps.append(float(parsing[5]))
                
                        huminitys.append((float(parsing[6])))
                        wind_directions.append((float(parsing[7])))
                        
                for num in range(len(stations)):
                    tmp_special_stations.append([huminitys[0],temps[0],wind_directions[0]])
                data_special_stations.append(tmp_special_stations)
                tmp_special_stations = []
data_special_stations = np.array(data_special_stations)
data_special_stations = np.reshape(data_special_stations,(-1,station_time,len(special_station_input_id)*3))
print("number of videos: ", data_special_stations.shape)

del tmp_special_stations
#data_special_stations = np.zeros((275,12,6))

# check if exist missing date in station data
print("[INFO] check missing data type")
if data_satellite.shape[0] ==0:
    print("missing satellite data, now pending...")
    data_satellite = np.zeros((data_station_huminity.shape[0],abs(satellite_frame),210,340,3))

if data_weathers.shape[0] == 0:
    print("missing weather data, now pending...")
    data_weathers = np.zeros((data_station_huminity.shape[0],1,340,210,3))


print("[INFO] train-test split")
# split trainning and testing data

(train_weather_X, test_weather_X, train_weather_Y, test_weather_Y, index_train, index_test) = train_test_split(data_weathers, category_labels, data_weather_labels,
	test_size=0.20, stratify=category_labels, random_state=random_st)

print("train_weather shape: ", train_weather_X.shape)
del data_weathers
print("finish split weather")

(train_temp_X, test_temp_X, train_temp_Y, test_temp_Y) = train_test_split(data_station_temperature, category_labels,
	test_size=0.20, stratify=category_labels, random_state=random_st)

del  train_temp_Y
del  test_temp_Y

del data_station_temperature
print("finish split temp")

(train_huminity_X, test_huminity_X, train_huminity_Y, test_huminity_Y) = train_test_split(data_station_huminity, category_labels,
	test_size=0.20, stratify=category_labels, random_state=random_st)

del train_huminity_Y
del test_huminity_Y

del data_station_huminity
print("finish split huminity")

(train_wind_X, test_wind_X, train_wind_Y, test_wind_Y) = train_test_split(data_station_wind_direction, category_labels,
	test_size=0.20, stratify=category_labels, random_state=random_st)

del train_wind_Y
del test_wind_Y

del data_station_wind_direction
print("finish split wind")

(train_satellite_X, test_satellite_X, train_satellite_Y, test_satellite_Y) = train_test_split(data_satellite, category_labels,
	test_size=0.20, stratify=category_labels, random_state=random_st)

del train_satellite_Y
del test_satellite_Y

del data_satellite
print("finish split satellite")

(train_special_stations_X, test_special_stations_X, train_special_stations_Y, test_special_stations_Y) = train_test_split(data_special_stations, category_labels,
	test_size=0.20, stratify=category_labels, random_state=random_st)

del train_special_stations_Y
del test_special_stations_Y

del data_special_stations
print("finish split satellite")


print("[INFO] load model")
# load the model


model = keras.models.load_model(model_path)
model.summary()



# == Provide average scores ==
second = str(tt.time())

f_log = open(second+".txt", "w")
print('------------------------------------------------------------------------')
f_log.write('------------------------------------------------------------------------')



# make predictions on the testing set
print("[INFO] evaluating network...")
f_log.write("[INFO] evaluating network...")
predIdxs = model.predict([test_weather_X, 
test_temp_X, 
test_huminity_X, 
test_wind_X, 
test_satellite_X,
test_special_stations_X], batch_size=BS)


# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)
# show a nicely formatted classification report
#print("test index of true")
#print(labels[np.where(np.argmax(test_weather_Y, axis=1)==1)])
#f_log(labels[np.where(np.argmax(test_weather_Y, axis=1)==1)])
print(classification_report(test_weather_Y.argmax(axis=1), predIdxs,
	target_names=['False', 'True']))

f_log.write(classification_report(test_weather_Y.argmax(axis=1), predIdxs,
	target_names=['False', 'True']))

# compute the confusion matrix and and use it to derive the raw
# accuracy, sensitivity, and specificity
cm = confusion_matrix(test_weather_Y.argmax(axis=1), predIdxs)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
# show the confusion matrix, accuracy, sensitivity, and specificity
print(cm)
cm_txt = np.array2string(cm)
f_log.write(cm_txt)

print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))

#f_log.write(cm.tostring())
f_log.write("acc: {:.4f}".format(acc))
f_log.write("sensitivity: {:.4f}".format(sensitivity))
f_log.write("specificity: {:.4f}".format(specificity))




f_log.close()

