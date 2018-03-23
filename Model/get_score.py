# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 21:55:56 2018

@author: Mathieu Daviet
"""

from utils import *
from sklearn.cluster import KMeans
from collections import Counter

X_people = get_saved_features("FaceRecongition_feature/X_train_people", flatten = True)
X_tracks = get_saved_features("FaceRecongition_feature/means/X_train_tracks_means.pkl")
y_people = get_saved_features("FaceRecongition_feature/y_train_people", flatten = True)
y_tracks = get_saved_features("FaceRecongition_feature/means/y_train_tracks_means.pkl")




[X_test, X_train, y_test, y_train, list_movie_test] = split_dataset(X_tracks, y_tracks, 0.4)

list_Tracks_test = get_list_tracks(y_test)
test_tracks_length = len(list_Tracks_test)

scoretot = 0
itera = 1

for idmovie in list_movie_test:
    list_tracks_movie = get_list_tracks_by_movie(idmovie, y_test)
    X_kmeans = []
    y_kmeans = []
    for index in list_tracks_movie:
        X_kmeans.append(X_test[index])
        y_kmeans.append(y_test[index])

    continuer = True
    nbclusturs = int(len(y_kmeans)/3)

    kmeans = KMeans(n_clusters=nbclusturs, random_state=0)
    kmeans.fit(X_kmeans)
    for k in range(nbclusturs):
        votes = []
        for i in range(len(y_kmeans)):
            if kmeans.labels_[i] == k:
                votes.append(get_closest_people(y_kmeans[i][1], X_test, y_test, X_train, y_train, X_people, y_people))
        c = max(Counter(votes), key=Counter(votes).get)

        for i in range(len(y_kmeans)):
            if kmeans.labels_[i] == k:
                right_person = get_people_from_track(y_kmeans[i][1], y_test)
                if int(c) == int(right_person):
                    scoretot +=1

    
score = scoretot/test_tracks_length
print(score)
#0.7887