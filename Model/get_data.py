# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 13:23:50 2018

@author: Mathieu Daviet
"""

import numpy as np
import pickle


def get_mean_data(test_data = False, mean_people = True):
    X_train = get_saved_features("FaceRecongition_feature/Means/X_train")
    y_train = get_saved_features("FaceRecongition_feature/Means/y_train")
    if mean_people:
        X_people = get_saved_features("FaceRecongition_feature/Means/X_people")
    else:
        X_people = get_people_data()
    
    if test_data:
        X_test = get_saved_features("FaceRecongition_feature/Means/X_test")
        return [X_train, X_people, X_test, y_train]
    else:
        return [X_train, X_people, y_train]
    
    
def get_compressed_data(test_data = False, compressed_people = True):
    X_train = get_saved_features("FaceRecongition_feature/Compressed/X_train")
    y_train = get_saved_features("FaceRecongition_feature/Compressed/y_train")
    if compressed_people:
        X_people = get_saved_features("FaceRecongition_feature/Compressed/X_people")
    else:
        X_people = get_people_data()
    
    if test_data:
        X_test = get_saved_features("FaceRecongition_feature/Compressed/X_test")
        return [X_train, X_people, X_test, y_train]
    else:
        return [X_train, X_people, y_train]
    

def get_train_data():
    X_train = get_saved_features("FaceRecongition_feature/X_train_tracks")
    y_train = get_saved_features("FaceRecongition_feature/y_train_tracks")
    [X_train, y_train] = transform_xy_brut_data(X_train, y_train)
    
    return [X_train, y_train]
    
def get_people_data():
    X_people = get_saved_features("FaceRecongition_feature/X_train_people")
    y_people = get_saved_features("FaceRecongition_feature/y_train_people")
    [X_people, y_useless] = transform_xy_brut_data(X_people, y_people)
    
    return X_people

def get_test_data():
    X_test = get_saved_features("FaceRecongition_feature/X_test")
    return X_test
    
def get_saved_features(path):
    with open(path, 'rb') as f:
        x = pickle.load(f)
    
    if len(x) == 50:
        features = x[0]
        for bash in x[1:]:
            features = np.concatenate((features, bash))
        features = np.asarray(features)
    else:
        features = np.asarray(x)
    return features

def transform_xy_brut_data(X, y):
    X_new = []
    y_new = []
    for i in range(len(y)):
        y_new.append(y[i][0])
    for i in range(len(X)):
        if y[i][1] == None:
            X_new.append([X[i],y[i][0]])
        else:
            X_new.append([X[i],y[i][1]])
    return [X_new, y_new]