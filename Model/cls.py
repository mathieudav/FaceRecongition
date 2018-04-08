# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 16:25:31 2018

@author: Mathieu Daviet
"""
import scipy
import numpy as np

def get_closest_people(inputs_features, X_train_train, y_train_train, X_people,k ,distance):
    
    k_nearest_points = np.zeros((k, 2))
    
    for input_features in inputs_features:
        for k in range(len(X_train_train)):
            dist = compute_distance(input_features, X_train_train[k][0], distance)
            for i in range(len(k_nearest_points)):
                if dist < k_nearest_points[i][1] or k_nearest_points[i][1] == 0:
                    k_nearest_points[i][1] = dist
                    k_nearest_points[i][0] = y_train_train[k]
                    break
    
        for l in range(len(X_people)):
            dist = compute_distance(input_features, X_people[l][0], distance)
            for i in range(len(k_nearest_points)):
                if dist < k_nearest_points[i][1] or k_nearest_points[i][1] == 0:
                    k_nearest_points[i][1] = dist
                    k_nearest_points[i][0] = X_people[l][1]
                    break
    list_k_closest_people = []
    for i in range(len(k_nearest_points)):
        list_k_closest_people.append(k_nearest_points[i][0])
        
    
    max_occ_people = 0
    list_best_people = []
    for people in list_k_closest_people:
        if list_k_closest_people.count(people)>max_occ_people:
            max_occ_people = list_k_closest_people.count(people)
            list_best_people = [people]
        elif list_k_closest_people.count(people) == max_occ_people:
            list_best_people.append(people)
    
    list_best_people = list(set(list_best_people))
    
    people = 0
    min_score = 0
    for best_people in list_best_people:
        best_score_people = 0
        for nearest_point in k_nearest_points:
            if nearest_point[0] == best_people:
                if best_score_people > nearest_point[1] or best_score_people == 0:
                    best_score_people = nearest_point[1]
                    
        if min_score == 0 or best_score_people < min_score:
            people = best_people
            min_score = best_score_people
    
    return people


def get_closest_people_2(input_features, X_train_train, y_train_train, X_people, list_people, ratio, distance):

    min_dist = 0
    best_people = 0
    for people in list_people:
        list_dist = []
        for k in range(len(X_train_train)):
            if people == y_train_train[k]:
                list_dist.append(compute_distance(input_features, X_train_train[k][0], distance))
        for x in X_people:
            if people == x[1]:
                list_dist.append(compute_distance(input_features, x[0], distance))
        
        list_dist = list(sorted(list_dist))
        list_dist = list_dist[:int(len(list_dist)*ratio)+1]
        mean_dist = sum(list_dist)/len(list_dist)
        if mean_dist<min_dist or min_dist == 0:
            min_dist = mean_dist
            best_people = people
        
    return best_people
        
def compute_distance(features_1, features_2, distance):
    if distance == "braycurtis":
        return scipy.spatial.distance.braycurtis(features_1, features_2)
    if distance == "canberra":
        return scipy.spatial.distance.canberra(features_1, features_2)
    if distance == "chebyshev":
        return scipy.spatial.distance.chebyshev(features_1, features_2)
    if distance == "cityblock":
        return scipy.spatial.distance.cityblock(features_1, features_2)
    if distance == "correlation":
        return scipy.spatial.distance.correlation(features_1, features_2)
    if distance == "cosine":
        return scipy.spatial.distance.cosine(features_1, features_2)
    if distance == "euclidean":
        return scipy.spatial.distance.euclidean(features_1, features_2)
    if distance == "sqeuclidean":
        return scipy.spatial.distance.sqeuclidean(features_1, features_2)
        