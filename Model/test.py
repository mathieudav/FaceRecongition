# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 22:20:55 2018

@author: Mathieu Daviet
"""
from utils import *

X_test = get_saved_features("FaceRecongition_feature/Means/X_train", flatten = False)

print(len(X_test))