# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 21:29:57 2018

@author: valentin
"""

import retinopathy_kit.config as cfg
import retinopathy_kit.models.general_net as gennet


def get_metadata():
    """
    Simple call to get_metadata and set name to _Resnet50:
    """ 
    meta = gennet.get_metadata()
    meta['Name'] = "Retinopathy_Resnet50"

    return meta
        

def build_model():
    """
    Simple call to Resnet50:
    """  
    return gennet.build_model('Resnet50', cfg.RPKIT_LabelsNum)
        

def predict_file(img_path):
    """
    Simple call to gennet.predict_file() using Resnet50
    :param img_path: image to classify, full path  
    :return: most probable label
    """
    return gennet.predict_file(img_path, 'Resnet50')


def predict_data(img):
    """
    Simple call to gennet.predict_data() using Resnet50
    """    
    return gennet.predict_data(img, 'Resnet50')


def predict_url(*args):
    """
    Simple call to gennet.predict_url()
    """    

    return gennet.predict_url(*args)
        

def train(nepochs=15):
    """
    Simple call to gennet.train() using Resnet50
    """ 

    return gennet.train(nepochs, 'Resnet50')
