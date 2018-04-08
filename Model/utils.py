# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 21:22:58 2018

@author: demo
"""
import pickle
import numpy as np

def get_saved_features(path, flatten = False):
    with open(path, 'rb') as f:
        x = pickle.load(f)
    if flatten:
        input = x[0]
        for bash in x[1:]:
            input = np.concatenate((input, bash))
        return input
    return x

def get_list_tracks(y):
    List = []
    for doublet in y:
        List.append(doublet[1])
    return list(set(List))


def get_list_movies(y):
    list_tracks = get_list_tracks(y)
    list_movies = []
    for idtrack in list_tracks:
        list_movies.append(idtrack[:idtrack.index('_')])
    list_movies = list(set(list_movies))
    return list_movies
        

def get_list_index_tracks(id_track, y):
    indexs = []
    for i in range(len(y)):
        if y[i][1] == id_track:
            indexs.append(i)
    return indexs

def get_list_tracks_by_movie(idmovie, y):
    indexs = []
    idmovienorm = idmovie+"_"
    for i in range(len(y)):
        if y[i][1][:len(idmovienorm)] == idmovienorm:
            indexs.append(i)
    return indexs

def get_people_from_track(id_track, y):
    for i in range(len(y)):
        if y[i][1] == id_track:
            return y[i][0]
        
    print("error")
    
def list_unique_people(y):
    list_people = []
    nb_zeros = 0
    for y_u in y:
        list_people.append(y_u[0])
        if y_u[0] == 0:
            nb_zeros+=1
    print(nb_zeros)
    return list(set(list_people))
    
def get_mean_features_track(id_track, X, y):
    indexs = get_list_index_tracks(id_track, y)
    features = [[0 for col in range(len(X[0]))] for row in range(len(indexs))]
    for i in range(len(indexs)):
        features[i] = X[indexs[i]]
    
    median_features = np.mean(features, axis=0)
    
    return median_features

def get_closest_people(idtrack, X_test, y_test, X_train, y_train, X_people, y_people):
    features = get_mean_features_track(idtrack, X_test, y_test)

    dist_min = np.linalg.norm(features-X_people[0])
    idx_min = 0
    pred = y_train[idx_min][0]
    for i in range(len(X_train)):
        dist = np.linalg.norm(features-X_train[i])
        if dist < dist_min:
            dist_min = dist
            idx_min = i
            pred = y_train[idx_min][0]
   
    for i in range(len(X_people)):
        dist = np.linalg.norm(features-X_people[i])
        if dist < dist_min:
            dist_min = dist
            idx_min = i
            pred = y_people[idx_min][0]
    return pred
    
def split_dataset(X, y, ratio):
    list_movies = get_list_movies(y)
    list_movie_test = list_movies[:len(list_movies) - int(len(list_movies)*ratio)]
    X_test = []
    X_train = []
    y_test = []
    y_train = []
    for i in range(len(X)):
        idtrack = y[i][1]
        if idtrack[:idtrack.index('_')] in list_movie_test:
            X_test.append(X[i])
            y_test.append(y[i])
        else:
            X_train.append(X[i])
            y_train.append(y[i])
            
    return [X_test, X_train, y_test, y_train, list_movie_test]