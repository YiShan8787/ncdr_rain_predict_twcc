#!/bin/bash

a=0

#while [ $a -lt 10 ]
while true
do
   echo $a
   a=`expr $a + 1`
   python train_predict_rain.py
done
