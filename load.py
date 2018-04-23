import sys
from numpy import shape
import csv

sys.path.append('..')

import numpy as np
import os
from time import time
from collections import Counter
import random


def load_wind_data_new():
    #data created on Oct 3rd, WA 20 wind farms, 7 years
    with open('data/real.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    rows = np.array(rows[0:736128], dtype=float)
    trX = []
    print(shape(rows))
    m = np.ndarray.max(rows)
    print("Maximum value of wind", m)
    print(shape(rows))
    for x in range(5):
        train = rows[:736128, x].reshape(-1, 576)
        train = train / 16

        # print(shape(train))
        if trX == []:
            trX = train
        else:
            trX = np.concatenate((trX, train), axis=0)
    print("Shape TrX", shape(trX))

    with open('data/sample_label.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    trY = np.array(rows, dtype=int)
    print("Label Y shape", shape(trY))

    with open('data/index.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        index = [row for row in reader]
    index=np.array(index, dtype=int)

    print(shape(index))

    trX2=trX[index[0:23560]]
    trY2=trY[index[0:23560]]
    trX2=trX2.reshape([-1,576])
    teX=trX[index[23560:]]
    teX = teX.reshape([-1, 576])
    teY=trY[index[23560:]]

    csvfile = file('trainingX.csv', 'wb')
    writer = csv.writer(csvfile)
    samples = np.array(trX2*16, dtype=float)
    writer.writerows(samples.reshape([-1, 576]))

    csvfile = file('trainingY.csv', 'wb')
    writer = csv.writer(csvfile)
    samples = np.array(trY2, dtype=float)
    writer.writerows(samples)

    csvfile = file('testingX.csv', 'wb')
    writer = csv.writer(csvfile)
    samples = np.array(teX*16, dtype=float)
    writer.writerows(samples.reshape([-1, 576]))

    csvfile = file('testingY.csv', 'wb')
    writer = csv.writer(csvfile)
    samples = np.array(teY, dtype=float)
    writer.writerows(samples)

    with open('data/forecast data/24_hour_ahead_full.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    rows = np.array(rows[0:736128], dtype=float)
    forecastX = []
    print(shape(rows))
    m = np.ndarray.max(rows)
    m=np.clip(m,0, 16.0)
    print("Maximum value of wind", m)
    print(shape(rows))
    for x in range(20):
        train = rows[:736128, x].reshape(-1, 576)
        train = train / 16

        # print(shape(train))
        if forecastX == []:
            forecastX = train
        else:
            forecastX = np.concatenate((forecastX, train), axis=0)
    print("Shape ForecastX", shape(forecastX))
    forecastX=forecastX[index[23560:]]
    forecastX = forecastX.reshape([-1, 576])
    return trX2, trY2, teX, teY, forecastX



def load_wind_data_spatial():
    with open('spatial.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    rows = np.array(rows, dtype=float)
    m = np.ndarray.max(rows)
    print("Maximum value of wind", m)
    rows=rows/16
    return rows


def load_wind_mask():
    with open('power789.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    rows = np.array(rows, dtype=float)
    trX = []
    i=0
    for x in range(34):
        train = rows[:104832, x].reshape(-1, 576)
        train = train / 16
        # print(shape(train))
        if trX == []:
            trX = train
        else:
            trX = np.concatenate((trX, train), axis=0)
    print(trX.shape[0])
    trX2=trX
    st_pt=np.zeros((trX.shape[0],1))
    #st_pt = np.random.random_integers(375)
    #trX2[i, st_pt:st_pt + 100] = 0.5
    # coco_changes, Xiaoze changes again
    for i in range(trX.shape[0]):
        st_pt[i] = np.random.random_integers(375)
        trX2[i, st_pt[i]:st_pt[i]+ 100]=0.5

    return trX2 , st_pt


def load_solar_data():
    with open('solar label.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    labels = np.array(rows, dtype=int)
    print(shape(labels))

    with open('solar_0722.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    rows = np.array(rows, dtype=float)
    rows=rows[:104832,:]
    print(shape(rows))
    trX = np.reshape(rows.T,(-1,576))
    print(shape(trX))
    m = np.ndarray.max(rows)
    print("maximum value of solar power", m)
    trY=np.tile(labels,(32,1))
    trX=trX/m
    return trX,trY




def load_solar_data_orig():
    with open('solar.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    rows = np.array(rows, dtype=float)
    trX = []
    print(shape(rows))
    m = np.ndarray.max(rows)
    print(m)
    for x in range(52):
        train = rows[:, x].reshape(-1, 576)
        train = train / m
        # print(shape(train))
        if trX == []:
            trX = train
        else:
            trX = np.concatenate((trX, train), axis=0)
    return trX,m
