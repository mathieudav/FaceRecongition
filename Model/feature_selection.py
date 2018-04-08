# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 18:25:26 2018

@author: Mathieu Daviet
"""
import pickle
import numpy as np
from utils import *

# Getting back the objects:
with open('diff_features_tracks.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    diff_features = pickle.load(f)
   
median = np.median(diff_features)
list_goodindex = []
for i in range(len(diff_features)):
    if diff_features[i] < median:
        list_goodindex.append(i)

X_test = get_saved_features("FaceRecongition_feature/means/X_test_means.pkl")
X_people = get_saved_features("FaceRecongition_feature/X_train_people", flatten = True)
X_tracks = get_saved_features("FaceRecongition_feature/means/X_train_tracks_means.pkl")

X_test_new = []
X_people_new = []
X_tracks_new = []

for i in range(len(X_test)):
    print(i)
    vec = []
    for k in range(len(X_test[i])):
        if k in list_goodindex:
            vec.append(X_test[i][k])
            
    X_test_new.append(vec)
    

with open('FaceRecongition_feature/selected_features/X_test.pkl', 'wb') as f:
    pickle.dump(X_test_new, f)


for i in range(len(X_people)):
    vec = []
    for k in range(len(X_people[i])):
        if k in list_goodindex:
            vec.append(X_people[i][k])
            
    X_people_new.append(vec)

for i in range(len(X_tracks)):
    vec = []
    for k in range(len(X_tracks[i])):
        if k in list_goodindex:
            vec.append(X_tracks[i][k])
            
    X_tracks_new.append(vec)
    
with open('FaceRecongition_feature/selected_features/X_people.pkl', 'wb') as f:
    pickle.dump(X_people_new, f)
    
with open('FaceRecongition_feature/selected_features/X_tracks.pkl', 'wb') as f:
    pickle.dump(X_tracks_new, f)