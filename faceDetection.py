# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""
import numpy as np
import cv2
from base import base
import time
import affine
class faceDetector:
    def __init__(self):
        
        self.face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')
        self.profile_cascade= cv2.CascadeClassifier('haar/haarcascade_profileface.xml')
        self.eye_cascade = cv2.CascadeClassifier('haar/haarcascade_eye.xml')
        self.faces=[]
        self.img=None
        self.gray=None
        self.nb_faces=0
        
    def detect_face(self,img):
        self.img=img
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.faces = self.face_cascade.detectMultiScale(self.gray, 1.3, 5)
        self.profile=self.profile_cascade.detectMultiScale(self.gray, 1.3, 5)
        
        self.nb_faces=len(self.faces)
        print(self.nb_faces)

    def get_faces(self):
        for (x,y,w,h) in self.faces:
#            cv2.rectangle(self.img,(x,y),(x+w,y+h),(255,0,0),2)
            self.img=self.img[x:x+w,y:y+h]
#            roi_gray = self.gray[y:y+h, x:x+w]
#            roi_color = self.img[y:y+h, x:x+w]
#            eyes = self.eye_cascade.detectMultiScale(roi_gray)
#            for (ex,ey,ew,eh) in eyes:
#                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
         
#        for (x,y,w,h) in self.profile:
#            cv2.rectangle(self.img,(x,y),(x+w,y+h),(0,255,255),2)
#            roi_gray = self.gray[y:y+h, x:x+w]
#            roi_color = self.img[y:y+h, x:x+w]
#            eyes = self.eye_cascade.detectMultiScale(roi_gray)
#            for (ex,ey,ew,eh) in eyes:
#                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                
        return self.img
        
if __name__=="__main__":
    test=base()

    list_img=test.web_images_from_people_id("334")
    fd=faceDetector()
#    faces=[]
    faces=np.zeros([len(list_img),224,224,3],dtype=np.uint8)  
    i=0
    for img in list_img[:]:
#        face=cv2.imread(img).astype('uint8') 
        fd.detect_face(img)
        imgf=fd.get_faces()
        imgf=cv2.resize(imgf,(224,224))
        faces[i]=imgf
        i=i+1
#        cv2.imshow('img',imgf)
#        cv2.waitKey(0)  & 0xFF
    face_affine=affine.face_alignment(faces)
    for img in face_affine[:]:
        cv2.imshow('affine',img)
        cv2.waitKey(0)  & 0xFF
    cv2.destroyAllWindows()
