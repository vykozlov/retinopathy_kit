# -*- coding: utf-8 -*-
from os import path

# identify basedir for the package
BASE_DIR = path.dirname(path.normpath(path.dirname(__file__)))
RPKIT_Storage = 'https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/'
RPKIT_DatasetURL = RPKIT_Storage + 'dogImages.zip'
RPKIT_DataDir = 'retinopathy'
RPKIT_LabelsTrain = path.join(BASE_DIR,'data','rp_trainLabels.csv')
RPKIT_LabelsFile = path.join(BASE_DIR,'data','rp_diagnosis.txt')
RPKIT_LabelsNum = 5


