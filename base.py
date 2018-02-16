#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 16:52:41 2018

@author: francois
"""
import numpy as np
import cv2
import skimage.io as skio
import pandas as pd
import os

#répertoire par défault
os.chdir("/home/francois/Documents/SIR/Challenge/")



#classe permettant d'exploiter la base d'images
class base():
    def __init__(self):
        self.people=pd.read_csv("input_training/people.csv",names=["id_people","name"])
        self.tracks=pd.read_csv("challenge_output_data_training_file_celebrity_identification_challenge.csv",sep=";",index_col=0)
        self.web_images=None
        self.track_images=None
        self.list_tracks=None
    
    
    #récupére les images web liées à l'id de la célébrité 
    def web_images_from_people_id(self,people_id):
        img_directory_path="input_training/people/{}".format(people_id)
        list_img_path=["/".join([img_directory_path,img]) for img in os.listdir(img_directory_path)]
        list_img=[cv2.imread(img_path,1) for img_path in list_img_path if img_path.split('.')[-1]=='jpg']
        
        self.web_images=list_img
        
        return self.web_images
    
    #récupére les images issues des séquences de film par id
    def track_images_from_track_id(self,track_id):
        img_directory_path="input_training/tracks/{}".format(track_id)
        list_img_path=["/".join([img_directory_path,img])  for img in os.listdir(img_directory_path)]
        list_img=[cv2.imread(img_path,1) for img_path in list_img_path if img_path.split('.')[-1]=='jpg']
        
        self.track_images=list_img
        
        return self.track_images    

    #récupére les tracks pour une célébrité donnée (renvoie la liste des tracks dans lesquels people_id est présent)
    def track_from_people_id(self,people_id):
        self.list_tracks=list(self.tracks.query('people_id=={}'.format(people_id)).index)
    
        return self.list_tracks

if __name__=="__main__":
    test=base()
    liste_track=test.track_from_people_id("1578")
    list_track_img=test.track_images_from_track_id('94_1')
    list_web_img=test.web_images_from_people_id('1582')
    