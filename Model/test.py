# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 22:20:55 2018

@author: demo
"""
from utils import *

X_test = get_saved_features("FaceRecongition_feature/out.pickle", flatten = True)

print(len(X_test[1]))