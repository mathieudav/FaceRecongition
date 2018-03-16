#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 16:52:41 2018

@author: francois
"""
import numpy as np
import cv2
import pandas as pd
import os
import default_path



#classe permettant d'exploiter la base d'images
class base():
    def __init__(self):
        self.people=pd.read_csv("input_training/people.csv",names=["id_people","name"])
        self.tracks=pd.read_csv("challenge_output_data_training_file_celebrity_identification_challenge.csv",sep=";")
        self.web_images=None
        self.track_images=None
        self.list_tracks=None
    
    
    #récupére les images web liées à l'id de la célébrité 
    def web_images_from_people_id(self,people_id):
        img_directory_path="input_training/people/{}".format(people_id)
        list_img_path=["/".join([img_directory_path,img]) for img in os.listdir(img_directory_path)  if img.split('.')[-1]=='jpg']
        #list_img=[cv2.imread(img_path,1) for img_path in list_img_path if img_path.split('.')[-1]=='jpg']
        
        self.web_images=list_img_path
        
        return self.web_images
    
    #récupére les images issues des séquences de film par id
    def track_images_from_track_id(self,track_id,training=True):
        if training:
            img_directory_path="input_training/tracks/{}".format(track_id)
        else:
            img_directory_path="input_testing/{}".format(track_id)

        list_img_path=["/".join([img_directory_path,img])  for img in os.listdir(img_directory_path) if img.split('.')[-1]=='jpg']
        #list_img=[cv2.imread(img_path,1) for img_path in list_img_path ]
        
        self.track_images=list_img_path
        return self.track_images    

    #récupére les tracks pour une célébrité donnée (renvoie la liste des tracks dans lesquels people_id est présent)
    def track_from_people_id(self,people_id):
        self.list_tracks=list(self.tracks.query('people_id=={}'.format(people_id)).index)
    
        return self.list_tracks

if __name__=="__main__":
    test=base()
    liste_track=test.track_from_people_id("1578")
    list_track_img=[cv2.imread(img_path,1) for img_path in test.track_images_from_track_id('96_10',False)]
    list_web_img=[cv2.imread(img_path,1) for img_path in test.web_images_from_people_id('1582')]
    
