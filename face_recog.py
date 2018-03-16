#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 16:29:39 2018

@author: yan
"""
import numpy as np
import cv2  
import dlib  
import math
import matplotlib as plt
def face_alignment(faces):  
    ''''' 
    faces: num * width * height * channels ,value = 0~255, dtype = np.uint8,  
    note: width must equal to height 
    '''  
    print(len(faces))  

    num = len(faces)  
      
    faces_aligned = np.zeros((len(faces),*faces[0].shape),dtype=np.uint8)  
    
    predictor_path = "./shape_predictor_68_face_landmarks.dat" # dlib提供的训练好的68个人脸关键点的模型，网上可以下  
    predictor = dlib.shape_predictor(predictor_path) # 用来预测关键点  
    for i in range(num):  
        img = faces[i]  
        rec = dlib.rectangle(0,0,img.shape[0],img.shape[1])  
        shape = predictor(np.uint8(img),rec) # 注意输入的必须是uint8类型  
#        print('shape '+shape)
#        order=[36,45,30,48,54] # left eye, right eye, nose, left mouth, right mouth  注意关键点的顺序，这个在网上可以找  
#        if show:  
#            plt.pyplot.figure()  
#            plt.pyplot.imshow(img,cmap='gray')  
#            for j in order:  
#                x = shape.part(j).x  
#                y = shape.part(j).y  
#                plt.pyplot.scatter(x,y) # 可以plot出来看看效果，这里我只plot5个点  
        eye_center =( (shape.part(36).x + shape.part(45).x) * 1./2, # 计算两眼的中心坐标  
                      (shape.part(36).y + shape.part(45).y) * 1./2)   
        dx = (shape.part(45).x - shape.part(36).x) # note: right - right  
        dy = (shape.part(45).y - shape.part(36).y)  
          
        angle = math.atan2(dy,dx) * 180. / math.pi # 计算角度  
#        print angle  
        
        RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1) # 计算仿射矩阵  
        RotImg = cv2.warpAffine(img, RotateMatrix, (img.shape[0], img.shape[1])) # 进行放射变换，即旋转  
        faces_aligned[i] = RotImg  
    return faces_aligned # uint8  
