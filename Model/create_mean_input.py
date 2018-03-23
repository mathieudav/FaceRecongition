# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 20:26:34 2018

@author: demo
"""

from utils import *
import pickle
import numpy as np

X_test = get_saved_features("FaceRecongition_feature/X_test.pickle", flatten = True)
y_test = get_saved_features("FaceRecongition_feature/y_test.pickle", flatten = True)

list_tracks = get_list_tracks(y_test)


X = [get_mean_features_track(list_tracks[0], X_test, y_test)]
y = [(get_people_from_track(list_tracks[0], y_test), list_tracks[0])]


for idtrack in list_tracks[1:]:
    X = np.append(X, [get_mean_features_track(idtrack, X_test, y_test)], axis=0)
    y.append((get_people_from_track(idtrack, y_test),idtrack))
    
    
with open('FaceRecongition_feature/means/X_test_means.pkl', 'wb') as f:
    pickle.dump(X, f)
    
with open('FaceRecongition_feature/means/y_test_means.pkl', 'wb') as f:
    pickle.dump(y, f)