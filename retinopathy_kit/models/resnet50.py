# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 21:29:57 2018

@author: valentin
"""
import argparse
import time
import retinopathy_kit.config as cfg
import retinopathy_kit.models.general_net as gennet


def get_metadata():
    """
    Simple call to get_metadata and set name to _Resnet50:
    """ 
    meta = gennet.get_metadata()
    meta['Name'] = "Retinopathy_Resnet50"

    return meta  

def prepare_data():
    """
    Simple call to gennet.prepare_data() using Resnet50 
    """
    gennet.prepare_data('Resnet50')

def predict_file(img_path, model):
    """
    Simple call to gennet.predict_file() using Resnet50
    :param img_path: image to classify, full path  
    :return: most probable label
    """
    return gennet.predict_file(img_path, 'Resnet50', model)


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
        

def predict_kaggle(test_path, file_list):
    """
    Simple call to gennet.predict_kaggle() using Resnet50
    """    
    return gennet.predict_kaggle(test_path, file_list, 'Resnet50')

def train(nepochs, model):
    """
    Simple call to gennet.train() using Resnet50
    """ 
    if model == 'cnn':
        return gennet.train_cnn(nepochs, 'Resnet50')
    elif model == 'logreg':
        return gennet.train_logreg('Resnet50')
    
def main():
   
    if args.method == 'get_metadata':
        get_metadata()       
    elif args.method == 'predict_file':
        prepare_data()
        predict_file(args.file, model=args.model)
    elif args.method == 'predict_data':
        prepare_data()
        predict_data(args.file)
    elif args.method == 'predict_url':
        prepare_data()
        predict_url(args.url)
    elif args.method == 'predict_kaggle':
        prepare_data()
        start = time.time()          
        predict_kaggle(args.test_path, args.test_files)
        print("Elapsed time:  ", time.time() - start)
    elif args.method == 'train':
        prepare_data()
        train(args.n_epochs, args.model)
    else:
        get_metadata()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare data')
    parser.add_argument('--method', type=str, default="get_metadata",
                        help='Method to use: get_metadata (default), \
                        predict_file, predict_data, predict_url, train')
    parser.add_argument('--model', type=str, default='cnn', help='Model to use: cnn (default), logreg')
    parser.add_argument('--file', type=str, help='File to do prediction on, full path')
    parser.add_argument('--url', type=str, help='URL with the image to do prediction on')
    parser.add_argument('--test_path', type=str, help='Path to test files, full path')    
    parser.add_argument('--test_files', type=str, help='File list (csv) with test files, full path')        
    parser.add_argument('--n_epochs', type=int, default=15, 
                        help='Number of epochs to train on')    
    args = parser.parse_args()         
    
    main()
    