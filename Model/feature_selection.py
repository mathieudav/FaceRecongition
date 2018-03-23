# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 18:25:26 2018

@author: Mathieu Daviet
"""
import pickle

# Getting back the objects:
with open('objs.pkl') as f:  # Python 3: open(..., 'rb')
    obj0, obj1, obj2 = pickle.load(f)