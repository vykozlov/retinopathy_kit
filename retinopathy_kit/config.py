# -*- coding: utf-8 -*-
from os import path

# identify basedir for the package
BASE_DIR = path.dirname(path.normpath(path.dirname(__file__)))
#RPKIT_Storage = 'https://nc.deep-hybrid-datacloud.eu/s/pTfMWcBjFE6qGJa'
RPKIT_Storage = 'deep-nextcloud:/Datasets/retinopathy/'
RPKIT_LabelsTrain = path.join(BASE_DIR,'data','trainLabels.csv')
RPKIT_Categories = path.join(BASE_DIR,'data','rp_diagnosis.txt')
RPKIT_LabelsNum = 5

