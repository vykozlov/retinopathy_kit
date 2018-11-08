# -*- coding: utf-8 -*-
"""
Feature building
"""

# import project config.py
import retinopathy_kit.config as cfg
import os
import numpy as np
import retinopathy_kit.dataset.data_utils as dutils
#from tqdm import tqdm

def set_feature_file(data_type, network = 'Resnet50'):
    """ Construct file name for the bottleneck file
    """
    return network + '_features_' + data_type + '.npz'

def build_features_set(img_files, set_type, network = 'Resnet50'):
    """Build bottleneck_features for set of files"""

    nets = {'VGG16': extract_VGG16,
            'VGG19': extract_VGG19,
            'Resnet50': extract_Resnet50,
            'InceptionV3': extract_InceptionV3,
            'Xception': extract_Xception,
    }
    
    bottleneck_features = nets[network](dutils.paths_to_tensor(img_files))

    bottleneck_file =  set_feature_file(set_type, network)
    bottleneck_path = os.path.join(cfg.BASE_DIR,'models',
                                   'bottleneck_features', bottleneck_file)

    if set_type == 'train':
        np.savez(bottleneck_path, train=bottleneck_features)
    elif set_type == 'test':
        np.savez(bottleneck_path, test=bottleneck_features)
    elif set_type == 'valid':
        np.savez(bottleneck_path, valid=bottleneck_features)
    else:
        np.savez(bottleneck_path, features=bottleneck_features)
    
    print("[INFO] Bottleneck features size (build_features):", bottleneck_features.shape)    
    
    return bottleneck_features


def check_features_set(data_type, network = 'Resnet50'):
    """Check if features file exists locally
       Only one dataset, e.g. train, valid, test is checked
    """
    bottleneck_file =  set_feature_file(data_type, network)
    bottleneck_exists, _ = dutils.maybe_download_data(
                                    data_dir='/models/bottleneck_features',
                                    data_file = bottleneck_file)        

    if not bottleneck_exists:
        print("[INFO] %s was neither found nor downloaded. Trying to build. It may take time .. " 
              % bottleneck_file)
        data_dir = os.path.join(cfg.BASE_DIR,'data', data_type)
        img_files, targets = dutils.load_dataset(data_dir)
        build_features_set(img_files, data_type, network)
        
        bottleneck_path = os.path.join(cfg.BASE_DIR,'models',
                                       'bottleneck_features', bottleneck_file)
        dest_dir = cfg.RPKIT_Storage.rstrip('/') + '/models/bottleneck_features'
        dutils.rclone_copy(bottleneck_path, dest_dir)

def load_features_set(data_type, network = 'Resnet50'):
    """Load features from the file
       Only one dataset, e.g. train, valid, test is loaded
    """
    bottleneck_file =  set_feature_file(data_type, network)
    bottleneck_path = os.path.join(cfg.BASE_DIR,'models',
                                   'bottleneck_features', bottleneck_file)
    print("[INFO] Using %s" % bottleneck_file)
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