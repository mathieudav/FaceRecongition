# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 14:44:18 2018

@author: Mathieu Daviet
"""
def split_train_data(X, y, ratio_train = 0.6):
    list_movies = get_list_movies(X)
    list_movies_train = list_movies[:int(len(list_movies)*ratio_train)]
    i = 0
    while(get_movie(X[i]) in list_movies_train):
        i+=1
        
    X_train_train = X[:i]
    y_train_train = y[:i]
    
    X_train_test = X[i:]
    y_train_test = y[i:]
    
    return [X_train_train, X_train_test, y_train_train, y_train_test]

def get_list_movies(X):
    list_movies = []
    for x in X:
        movieid = get_movie(x)
        list_movies.append(movieid)
    
    list_movies = list(sorted(list(set(list_movies))))
    
    return list_movies
    
def get_movie(x):
    idtrack = x[1]
    return idtrack[:idtrack.find('_')]
    
    
def get_list_tracks(X):
    list_tracks = []
    for x in X:
        list_tracks.append(x[1])
    list_tracks = list(sorted(list(set(list_tracks))))
    return list_tracks

def get_list_people_from_people_and_tracks(X_people, y_tracks):
    list_people = []
    for x in X_people:
        list_people.append(x[1])
    list_people = list(set(list_people)) 
    for people in y_tracks:
        list_people.append(people)
    list_people = list(set(list_people))
    return list_people