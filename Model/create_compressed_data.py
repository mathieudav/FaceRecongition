# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 10:15:35 2018

@author: Mathieu Daviet
"""

import numpy as np
import get_data as gd
import manage_data as md
import pickle
import math

def create_meadian_input(type_data, nb_vectors_track):
    if type_data == "test":
        X = gd.get_test_data()
        y = []
    if type_data == "people":
        X = gd.get_people_data()
        y = []
    if type_data == "train":
        [X, y] = gd.get_train_data()
        
    list_tracks = md.get_list_tracks(X)
    X_new = []
    y_new = np.zeros(len(list_tracks)*nb_vectors_track, dtype=int)
    for t in range(len(list_tracks)):
        X_mean = []
        for i in range(len(X)):
            if X[i][1] == list_tracks[t]:
                if len(X_mean) == 0:
                    X_mean = [X[i][0]]
                else:
                    X_mean = np.concatenate((X_mean, [X[i][0]]))
                if len(y)>1:
                    y_val  = int(y[i])
        
        lenXmean = len(X_mean)
        for i in range(nb_vectors_track):
            if i<=int(lenXmean/nb_vectors_track):
                X_med_temp = np.median(X_mean[math.ceil(lenXmean*i/nb_vectors_track):math.ceil(lenXmean*(i+1)/nb_vectors_track)], axis = 0)
                if len(X_new) == 0:
                    X_new = [[X_med_temp, list_tracks[t]]]
                else:
                    if len(y)>1:
                        y_new[len(X_new)] = y_val
                    if len(X_new)>0:
                        X_new.append([X_med_temp, list_tracks[t]])
    if type_data == "test":
        with open('FaceRecongition_feature/Compressed/X_test', 'wb') as f:
            pickle.dump(X_new, f)
    if type_data == "people":
        with open('FaceRecongition_feature/Compressed/X_people', 'wb') as f:
            pickle.dump(X_new, f)
    if type_data == "train":
        with open('FaceRecongition_feature/Compressed/X_train', 'wb') as f:
            pickle.dump(X_new, f)
        with open('FaceRecongition_feature/Compressed/y_train', 'wb') as f:
            pickle.dump(y_new, f)

create_meadian_input("test", 8)
create_meadian_input("people", 8)
create_meadian_input("train", 8)
