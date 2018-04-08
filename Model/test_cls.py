# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 16:27:59 2018

@author: Mathieu Daviet
"""

import get_data as gd
import manage_data as md
import cls

[X_train, X_people, y_train] = gd.get_compressed_data(test_data = False, compressed_people = False)
[X_train, X_train_test, y_train, y_train_test] = md.split_train_data(X_train, y_train, 0.6)

#list_people = md.get_list_people_from_people_and_tracks(X_people, y_train_train)
len_test = len(X_train_test)

score = 0
current_track = X_train_test[0][1]
imput_features = [X_train_test[0][0]]
for i in range(len_test):
    if current_track == X_train_test[i][1]:
        imput_features.append(X_train_test[i][0])
    else:
        print(str(i) + "/" + str(len_test))
        closest_people = cls.get_closest_people(imput_features, X_train, y_train, X_people, 4, distance = "euclidean")
        if int(closest_people) == int(y_train_test[i-1]):
            score+=1
        imput_features = [X_train_test[i][0]]
        current_track = X_train_test[i][1]

print(score/len(md.get_list_tracks(X_train_test)))