# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""
import numpy as np
import math
from scipy.io import loadmat
import os
os.environ["THEANO_FLAGS"]="device=cuda0,floatX=float32"
os.environ["KERAS_BACKEND"]="tensorflow"

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Dropout, Activation, Lambda, Permute, Reshape
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras import backend as K
K.set_image_data_format( 'channels_last' ) # WARNING : important for images and tensors dimensions ordering

        
def convblock(cdim, nb, bits=3):
    L = []
    
    for k in range(1,bits+1):
        convname = 'conv'+str(nb)+'_'+str(k)

        L.append( Convolution2D(cdim, kernel_size=(3, 3), padding='same', activation='relu', name=convname) ) 
    
    L.append( MaxPooling2D((2, 2), strides=(2, 2)) )
    
    return L

def copy_mat_to_keras(kmodel,data):
    l = data['layers']

    kerasnames = [lr.name for lr in kmodel.layers]

    # WARNING : important setting as 2 of the 4 axis have same size dimension
    #prmt = (3,2,0,1) # INFO : for 'th' setting of 'dim_ordering'
    prmt = (0,1,2,3) # INFO : for 'channels_last' setting of 'image_data_format'

    for i in range(l.shape[1]):
        matname = l[0,i][0,0].name[0]
        if matname in kerasnames:
            kindex = kerasnames.index(matname)
            #print matname
            l_weights = l[0,i][0,0].weights[0,0]
            l_bias = l[0,i][0,0].weights[0,1]
            f_l_weights = l_weights.transpose(prmt)
            #f_l_weights = np.flip(f_l_weights, 2) # INFO : for 'th' setting in dim_ordering
            #f_l_weights = np.flip(f_l_weights, 3) # INFO : for 'th' setting in dim_ordering
            assert (f_l_weights.shape == kmodel.layers[kindex].get_weights()[0].shape)
            assert (l_bias.shape[1] == 1)
            assert (l_bias[:,0].shape == kmodel.layers[kindex].get_weights()[1].shape)
            assert (len(kmodel.layers[kindex].get_weights()) == 2)
            kmodel.layers[kindex].set_weights([f_l_weights, l_bias[:,0]])
            #print '------------------------------------------'



class vgg:
    def __init__(self):
        
        model=self.__get_vgg_model()
        data = loadmat('vgg-face.mat', matlab_compatible=False, struct_as_record=False)
        copy_mat_to_keras(model,data)
        self.featureModel=Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
        
    def get_img_features(self,list_crpimg):    
        list_imarr = [np.array(crpimg).astype(np.float32) for crpimg in list_crpimg]
    
        #list_imarr = [np.expand_dims(imarr, axis=0) for imarr in list_imarr]
        
    
        list_fvec =self.featureModel.predict(np.array(list_imarr),batch_size=128,verbose=1)
        
        return list_fvec
        
    def __get_vgg_model(self):
        mdl = Sequential()
        
        mdl.add( Permute((1,2,3), input_shape=(224,224,3)) )

        for l in convblock(64, 1, bits=2):
            mdl.add(l)

        for l in convblock(128, 2, bits=2):
            mdl.add(l)
        
        for l in convblock(256, 3, bits=3):
            mdl.add(l)
            
        for l in convblock(512, 4, bits=3):
            mdl.add(l)
            
        for l in convblock(512, 5, bits=3):
            mdl.add(l)
        
        mdl.add( Convolution2D(4096, kernel_size=(7, 7), activation='relu', name='fc6') )
        mdl.add( Dropout(0.5) )
        mdl.add( Convolution2D(4096, kernel_size=(1, 1), activation='relu', name='fc7') )
        mdl.add( Dropout(0.5) )

        mdl.add( Convolution2D(2622, kernel_size=(1, 1), activation='relu', name='fc8') )
        mdl.add( Flatten() )
        mdl.add( Activation('softmax') )
        
        return mdl
        
        

        
    

if __name__=="__main__":
    import base
    import cv2
    
    base_img=base.base()
    vgg_test=vgg()
    
    
    list_img=[cv2.imread(img_path,1) for img_path in base_img.web_images_from_people_id("337")]
    list_img=[cv2.resize(img,(224,224)) for img in list_img]
    list_img_features=[]
    for img in list_img[:5]:
        cv2.imshow('img',img)
        cv2.waitKey(0)  & 0xFF
    list_img_features=vgg_test.get_img_features(list_img)
    cv2.destroyAllWindows()

    
    
    
