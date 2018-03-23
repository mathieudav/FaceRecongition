# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 15:53:45 2018

@author: Mathieu Daviet
"""

from utils import *
from sklearn.cluster import KMeans

X_people = get_saved_features("FaceRecongition_feature/X_train_people", flatten = True)
X_tracks = get_saved_features("FaceRecongition_feature/means/X_train_tracks_means.pkl")
y_people = get_saved_features("FaceRecongition_feature/y_train_people", flatten = True)
y_tracks = get_saved_features("FaceRecongition_feature/means/y_train_tracks_means.pkl")



[X_test, X_train, y_test, y_train, list_movie_test] = split_dataset(X_tracks, y_tracks, ratio = 0)



for idmovie in [list_movie_test[0]]:
    list_tracks_movie = get_list_tracks_by_movie(idmovie, y_test)
    X_kmeans = []
    y_kmeans = []
    for index in list_tracks_movie:
        X_kmeans.append(X_test[index])
        y_kmeans.append(y_test[index])
        
    print(len(y_kmeans))
    
    continuer = True
    nbclusturs = 1
    score =0
    while(continuer):
        nbclusturs += 1
        kmeans = KMeans(n_clusters=nbclusturs, random_state=0)
        kmeans.fit(X_kmeans)
        print(str(nbclusturs)+ " : "+str(kmeans.score(X_kmeans)))
        if (score < kmeans.score(X_kmeans) or score==0):
            score = kmeans.score(X_kmeans)
        else:
            continuer = False
    kmeans = KMeans(n_clusters=nbclusturs-1, random_state=0)
    kmeans.fit(X_kmeans)
    for k in range(nbclusturs):    
        for i in range(len(y_kmeans)):
            if kmeans.labels_[i] == k:
                print(str(kmeans.labels_[i])+":"+str(y_kmeans[i][0]))