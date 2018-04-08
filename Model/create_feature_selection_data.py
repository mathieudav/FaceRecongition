# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 21:08:56 2018

@author: Mathieu Daviet
"""

from utils import *
import numpy as np
import pickle

#[X_people] = get_saved_features("FaceRecongition_feature/scaled/X_train_people_scaled.pkl")
X_tracks = get_saved_features("FaceRecongition_feature/means/X_train_tracks_means.pkl")
#y_people = get_saved_features("FaceRecongition_feature/y_train_people", flatten = True)
y_tracks = get_saved_features("FaceRecongition_feature/means/y_train_tracks_means.pkl")

list_Tracks = get_list_tracks(y_tracks)
test_tracks_length = len(list_Tracks)

print(len(X_tracks))

diff_features = [0.]*len(X_tracks[0])

for i in range(len(X_tracks)):
    print(i)
    for j in range(len(X_tracks)):
        if int(y_tracks[i][0]) == int(y_tracks[j][0]):
            if i != j:
                diff_features += np.absolute(X_tracks[i] - X_tracks[j])
            
            
with open('diff_features_tracks.pkl', 'wb') as f:
    pickle.dump(diff_features, f)