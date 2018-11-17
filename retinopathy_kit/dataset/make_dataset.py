# -*- coding: utf-8 -*-
# import project config.py
"""
"""
import os
import logging
import argparse
from pathlib2 import Path
from dotenv import find_dotenv, load_dotenv
import retinopathy_kit.config as cfg
import retinopathy_kit.dataset.data_utils as dutils
import retinopathy_kit.features.build_features as bfeatures


def check_targets(data_type, remote_storage=cfg.RPKIT_Storage):
    """Check if targets file exists locally
       Only one dataset, e.g. train, valid, test is checked
    """
    data_dir = '/data'
    targets_file = 'RPKIT_targets_' + data_type + '.npz'
    targets_path = os.path.join(cfg.BASE_DIR, 'data', targets_file)
    targets_exists, _ = dutils.maybe_download_data(
                                    data_dir=data_dir,
                                    data_file = targets_file)        

    if not targets_exists:
        print("[INFO] %s was neither found nor downloaded. Trying to build .. " 
              % targets_file)
        # check if directory with train, test, and valid images exists:
        dutils.maybe_download_and_unzip()
        dutils.build_targets(data_type)
        # Upload to nextcloud newly created file
        targets_exists = True if os.path.exists(targets_path) else False
        dest_dir = remote_storage.rstrip('/') + data_dir
        print("[INFO] Upload %s to %s" % (targets_path, dest_dir))        
        dutils.rclone_copy(targets_path, dest_dir)
        
    return targets_exists

def check_features(data_type, network = 'Resnet50', remote_storage=cfg.RPKIT_Storage):
    """Check if features file exists locally
       Only one dataset, e.g. train, valid, test is checked
    """
    bottleneck_file =  bfeatures.set_feature_file(data_type, network)
    bottleneck_path = os.path.join(cfg.BASE_DIR,'data',
                                   'bottleneck_features', bottleneck_file)
    bottleneck_exists, _ = dutils.maybe_download_data(
                                    data_dir='/data/bottleneck_features',
                                    data_file = bottleneck_file)        

    if not bottleneck_exists:
        print("[INFO] %s was neither found nor downloaded. Trying to build. It may take time .. " 
              % bottleneck_file)

        # check if directory with train, test, and valid images exists:
        dutils.maybe_download_and_unzip()
        bfeatures.build_features_set(data_type, network)
        
        # Upload to nextcloud newly created file
        bottleneck_exists = True if os.path.exists(bottleneck_path) else False                                       
        dest_dir = remote_storage.rstrip('/') + '/data/bottleneck_features'
        print("[INFO] Upload %s to %s" % (bottleneck_path, dest_dir))        
        dutils.rclone_copy(bottleneck_path, dest_dir)
        
    return bottleneck_exists

def prepare_data(network='Resnet50'):
    """ Function to prepare data
    """
   
    # check if rp_diagnosis file exists locally, if not -> download,
    # if not downloaded -> dutils.categories_create()
    rpkit_categories_file = cfg.RPKIT_Categories.split('/')[-1]
    status_rpkit_categories, _ = dutils.maybe_download_data(data_dir='/data', 
                                                     data_file=rpkit_categories_file)

    if not  status_rpkit_categories:
        print("[INFO] %s was neither found nor downloaded. Trying to create " 
              % rpkit_categories_file)
        dutils.categories_create()
    else:
        print("[INFO] %s exists" % (cfg.RPKIT_Categories))
                                                        
    # check if bottleneck_features file fexists locally
    # if not -> download it, if not downloaded -> try to build
    # train
    status = { True: "exists", False: "does not exist"}
    datasets = ['train', 'valid', 'test']
    for dset in datasets:
        status_targets = check_targets(dset)
        print("[INFO] Targets file for %s %s" % 
               (dset, status[status_targets]))        
        status_bottleneck = check_features(dset, network)
        print("[INFO] Bottleneck file for %s (%s) %s" % 
               (dset, network, status[status_bottleneck]))


def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    print("__%s__" % (args.network))
    prepare_data(args.network)    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    
    parser = argparse.ArgumentParser(description='Model parameters')
    parser.add_argument('--network', type=str, default="Resnet50",
                        help='Neural network to use: Resnet50, InceptionV3,\
                        VGG16, VGG19, Xception')
    args = parser.parse_args()
    

    main()
