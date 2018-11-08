# -*- coding: utf-8 -*-
"""
Feature building
"""

# import project config.py
import retinopathy_kit.config as cfg
import os
import numpy as np
import retinopathy_kit.dataset.data_utils as dutils
from tqdm import tqdm
   

def maybe_download_bottleneck(bottleneck_storage = cfg.RPKIT_Storage, 
                              bottleneck_file = 'Resnet50_features_train.npz'):
    """
    Download bottleneck features if they do not exist locally.
    :param bottleneck_file: name of the file to download
    """

    bottleneck_dir = os.path.join(cfg.BASE_DIR,'models','bottleneck_features')
    if not os.path.exists(bottleneck_dir):
        os.makedirs(bottleneck_dir)

    bottleneck_url = bottleneck_storage.rstrip('/') + \
                     os.path.join('/models/bottleneck_features', bottleneck_file)

    print("Bottleneck_url: ", bottleneck_url)

    bottleneck_path = os.path.join(bottleneck_dir, bottleneck_file)    
    # if bottleneck_features file does not exist, download it
    if not os.path.exists(bottleneck_path):
        status, _ = dutils.rclone_copy(bottleneck_url, bottleneck_dir)
        
def build_features(img_files, set_type, network = 'Resnet50'):
    """Build bottleneck_features for set of files"""

    nets = {'VGG16': extract_VGG16,
            'VGG19': extract_VGG19,
            'Resnet50': extract_Resnet50,
            'InceptionV3': extract_InceptionV3,
            'Xception': extract_Xception,
    }
    
    bottleneck_features = nets[network](dutils.paths_to_tensor(img_files))
    np.savez(os.path.join(cfg.BASE_DIR, 'models', 'bottleneck_features', 
             network + 'features_' + set_type), set_type=bottleneck_features)  
    
    print("Bottleneck features size (build_features):", bottleneck_features.shape)    
    
    return bottleneck_features

    
def load_features_set(data_type, network = 'Resnet50'):
    """Load features from the file
       Only one dataset, e.g. train, valid, test is loaded
    """

    bottleneck_file = network + '_features_' + data_type + '.npz'
    maybe_download_bottleneck(cfg.RPKIT_Storage, bottleneck_file)
    
    bottleneck_path = os.path.join(cfg.BASE_DIR,'models','bottleneck_features', bottleneck_file)
    bottleneck_features = np.load(bottleneck_path)[data_type]

    return bottleneck_features

def extract_VGG16(tensor):
	from keras.applications.vgg16 import VGG16, preprocess_input
	return VGG16(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def extract_VGG19(tensor):
	from keras.applications.vgg19 import VGG19, preprocess_input
	return VGG19(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def extract_Resnet50(tensor):
	from keras.applications.resnet50 import ResNet50, preprocess_input
	return ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def extract_Xception(tensor):
	from keras.applications.xception import Xception, preprocess_input
	return Xception(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def extract_InceptionV3(tensor):
	from keras.applications.inception_v3 import InceptionV3, preprocess_input
	return InceptionV3(weights='imagenet', include_top=False).predict(preprocess_input(tensor))