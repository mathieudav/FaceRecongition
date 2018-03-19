# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""
#environ 100 000 images
number_of_batch=50

import base
import faceDetection
import face_recog

import numpy as np
import cv2
import vgg
import os 

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from sklearn.cluster import KMeans
from time import time

from multiprocessing import Pool
import pickle

base_img=base.base()
fd=faceDetection.faceDetector()
vgg_test=vgg.vgg()

def load_preprocess_data(tup):
    img_path=tup[0]
    ids=tup[1]
    img=cv2.imread(img_path,1)
#    fd.detect_face(img)
#    imgf=fd.get_faces()
    
#    if imgf.any():
#        #print(" pas de faces détectées : {} ".format(img_path))
#        return
#    else:
#        resized_img=cv2.resize(img,(224,224))
#        face_affine=face_recog.face_alignment(resized_img)[0]
#        
#        return (face_affine ,ids)
     
    resized_img=cv2.resize(img,(224,224))
    return (resized_img ,ids)

def split_array(array,nb_split):
    n=len(array)
    l=int(n/nb_split+0.5)
    return [array[i*l:(i+1)*l] if i<nb_split-1 else array[i*l:] for i in range(nb_split)]





t0 = time()


training_data_set_tracks=[(img_path,(people_id,track_id)) for track_id,people_id in base_img.tracks.values if people_id!=None for img_path in base_img.track_images_from_track_id(track_id) if img_path!=None ]
X_train_tracks=[]
y_train_tracks=[]
i=len(training_data_set_tracks)
for batch in split_array(training_data_set_tracks,number_of_batch):
    training_set=[]
    with Pool(8) as p:
        training_set=p.map(load_preprocess_data,batch)
    print(i)
    i-=len(batch)
    l=np.array(training_set).transpose()
    X_train_tracks.append(vgg_test.get_img_features(l[0]))
    y_train_tracks.append(l[1])

t1=time() 
print("Loaded and computed features on training data in  %0.3fs" % (t1 - t0))

training_data_set_people=[(img_path,(people_id,None)) for people_id in os.listdir("input_training/people") if people_id!=None for img_path in base_img.web_images_from_people_id(people_id) if img_path!=None ]
X_train_people=[]
y_train_people=[]
i=len(training_data_set_people)
for batch in split_array(training_data_set_people,number_of_batch):
    training_set=[]
    with Pool(8) as p:
        training_set=p.map(load_preprocess_data,batch)
    print(i)
    i-=len(batch)
    l=np.array(training_set).transpose()
    X_train_people.append(vgg_test.get_img_features(l[0]))
    y_train_people.append(l[1])

t2=time() 
print("Loaded and computed features on training data in  %0.3fs" % (t2 - t1))

test_data_set=[(img_path,track_id) for track_id in os.listdir("input_testing") for img_path in base_img.track_images_from_track_id(track_id,False) if img_path!=None ]
X_test=[]

i=len(test_data_set)
for batch in split_array(test_data_set,number_of_batch):
    test_set=[]
    with Pool(8) as p:
        test_set=p.map(load_preprocess_data,batch)
    print(i)
    i-=len(batch)
    l=np.array(test_set).transpose()
    X_test.append(vgg_test.get_img_features(l[0]))
    
t3=time()
print("Loaded and computed features on training data in  %0.3fs" % (t3 - t2))


with open("X_train_tracks.pickle","wb") as f:
    pickle.dump(X_train_tracks,f)
with open("y_train_tracks.pickle","wb") as f:
    pickle.dump(y_train_tracks,f)
with open("X_train_people.pickle","wb") as f:
    pickle.dump(X_train_people,f)
with open("y_train_people.pickle","wb") as f:
    pickle.dump(y_train_people,f)
with open("X_test.pickle","wb") as f:
    pickle.dump(X_test,f)
with open("X_test.pickle","wb") as f:
    pickle.dump(X_test,f)

"""
t0 = time()


print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)


print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
"""
