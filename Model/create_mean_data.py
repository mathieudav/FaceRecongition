# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 20:26:34 2018

@author: Mathieu Daviet
"""
import numpy as np
import get_data as gd
import manage_data as md
import pickle

def create_mean_input(type_data):
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
    y_new = np.zeros(len(list_tracks), dtype=int)
    for t in range(len(list_tracks)):
        X_mean = []
        for i in range(len(X)):
            if X[i][1] == list_tracks[t]:
                if len(X_mean) == 0:
                    X_mean = [X[i][0]]
                else:
                    X_mean = np.concatenate((X_mean, [X[i][0]]))
                if len(y)>1:
                    y_new[t] = int(y[i])
        
        X_mean = np.mean(X_mean, axis = 0)
        if len(X_new) == 0:
            X_new = [[X_mean, list_tracks[t]]]
        else:
            X_new.append([X_mean, list_tracks[t]])
    if type_data == "test":
        with open('FaceRecongition_feature/Means/X_test', 'wb') as f:
            pickle.dump(X_new, f)
    if type_data == "people":
        with open('FaceRecongition_feature/Means/X_people', 'wb') as f:
            pickle.dump(X_new, f)
    if type_data == "train":
        with open('FaceRecongition_feature/Means/X_train', 'wb') as f:
            pickle.dump(X_new, f)
        with open('FaceRecongition_feature/Means/y_train', 'wb') as f:
            pickle.dump(y_new, f)


create_mean_input("test")
create_mean_input("people")
create_mean_input("train")