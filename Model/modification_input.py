# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 14:08:50 2018

@author: demo
"""

from utils import *
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

X_people = get_saved_features("FaceRecongition_feature/X_train_people", flatten = True)
X_tracks = get_saved_features("FaceRecongition_feature/X_train_tracks", flatten = True)
X_test = get_saved_features("FaceRecongition_feature/X_test.pickle", flatten = True)


#Normalize data
scaler = StandardScaler()
scaler.fit(X_test)

X_people = scaler.transform(X_people)
X_tracks = scaler.transform(X_tracks)
X_test = scaler.transform(X_test)

print("ok")

with open('FaceRecongition_feature/X_train_people_scaled.pkl', 'wb') as f:
    pickle.dump([X_people], f)
    
with open('FaceRecongition_feature/X_train_tracks_scaled.pkl', 'wb') as f:
    pickle.dump([X_tracks], f)
    
with open('FaceRecongition_feature/X_test_scaled.pkl', 'wb') as f:
    pickle.dump([X_test], f)