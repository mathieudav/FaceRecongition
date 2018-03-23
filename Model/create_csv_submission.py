# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 21:49:49 2018

@author: Mathieu Daviet
"""


from utils import *

X_people = get_saved_features("FaceRecongition_feature/X_train_people", flatten = True)
X_tracks = get_saved_features("FaceRecongition_feature/means/X_train_tracks_means.pkl")
X_test = get_saved_features("FaceRecongition_feature/means/X_test_means.pkl")
y_people = get_saved_features("FaceRecongition_feature/y_train_people", flatten = True)
y_tracks = get_saved_features("FaceRecongition_feature/means/y_train_tracks_means.pkl")
y_test = get_saved_features("FaceRecongition_feature/means/y_test_means.pkl")

list_Tracks_test = list(sorted(get_list_tracks(y_test)))

test_tracks_length = len(list_Tracks_test)

itera = 1
save_lines = ["[track_id],[people_id]"]
for idtrack in list_Tracks_test:
    print(str(itera) + "/" + str(test_tracks_length))
    itera += 1
    closest_person = get_closest_people(idtrack, X_test, y_test, X_tracks, y_tracks, X_people, y_people)
    save_lines.append(','.join([str(idtrack), str(closest_person)]))
    
#Sauvegarde du fichier
with open('submit_file.csv', 'w') as f:
    f.write('\n'.join(save_lines))