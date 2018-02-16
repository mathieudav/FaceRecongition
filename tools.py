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

#set default folder
os.chdir("/home/francois/Documents/SIR/Challenge/")


class base():
    def __init__(self):
        self.people=pd.read_csv("input_training/people.csv",names=["id_people","name"])
        self.tracks=pd.read_csv("challenge_output_data_training_file_celebrity_identification_challenge.csv",names=["id_track","id_people"],sep=";")
        self.web_images=None
        self.track_images=None

    def web_images_from_people_id(self,people_id):
        img_directory_path="input_training/people/{}".format(people_id)
        list_img_path=["/".join([img_directory_path,img]) for img in os.listdir(img_directory_path)]
        list_img=[cv2.imread(img_path,1) for img_path in list_img_path if img_path.split('.')[-1]=='jpg']
        
        self.web_images=list_img
        
        return self.web_images

    def track_images_from_track_id(self,track_id):
        img_directory_path="input_training/tracks/{}".format(track_id)
        list_img_path=["/".join([img_directory_path,img])  for img in os.listdir(img_directory_path)]
        list_img=[cv2.imread(img_path,1) for img_path in list_img_path if img_path.split('.')[-1]=='jpg']
        
        self.track_images=list_img
        
        return self.track_images    
    
    def track_images_from_people_id(self,people_id):
        img_directory_path="input_training/tracks/{}".format(people_id)
        list_img_path=["/".join([img_directory_path,img])  for img in os.listdir(img_directory_path)]
        list_img=[cv2.imread(img_path,1) for img_path in list_img_path if img_path.split('.')[-1]=='jpg']
        
        self.track_images=list_img
        
        return self.track_images

if __name__=="__main__":
    test=base()
    img=test.track_images_from_track_id("94_1")