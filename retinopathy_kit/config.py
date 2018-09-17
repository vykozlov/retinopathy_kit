# -*- coding: utf-8 -*-
from os import path

# identify basedir for the package
BASE_DIR = path.dirname(path.normpath(path.dirname(__file__)))
RPKIT_Storage = 'https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/'
RPKIT_DatasetURL = RPKIT_Storage + 'dogImages.zip'
RPKIT_DataDir = 'dogImages'
RPKIT_LabelsFile = path.join(BASE_DIR,'data','dog_names.txt')
RPKIT_LabelsNum = 133


