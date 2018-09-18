# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 21:29:57 2018

@author: valentin
"""

import os
import tempfile
import numpy as np
import pkg_resources
import werkzeug.exceptions as exceptions
import retinopathy_kit.config as cfg
import retinopathy_kit.dataset.data_utils as dutils
import retinopathy_kit.models.model_utils as mutils
import retinopathy_kit.features.build_features as bfeatures
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras import backend


def get_metadata():
    
    module = __name__.split('.', 1)    
    
    pkg = pkg_resources.get_distribution(module[0])
    meta = {
        'Name': None,
        'Version': None,
        'Summary': None,
        'Home-page': None,
        'Author': None,
        'Author-email': None,
        'License': None,
    }

    for l in pkg.get_metadata_lines("PKG-INFO"):
        for par in meta:
            if l.startswith(par):
                k, v = l.split(": ", 1)
                meta[par] = v
                
    return meta
        

def build_model(network='Resnet50', nclasses=cfg.RPKIT_LabelsNum):
    """
    Build network. Possible nets:
    Resnet50, VGG19, VGG16, InceptionV3, Xception
    """
    
    train_net = bfeatures.load_features_set('train', network)
    #train_net, _, _ = bfeatures.load_features_all(network)
    # introduce bottleneck_features shapes manually 
    #features_shape = {'VGG16': [7, 7, 512],
    #                  'VGG19': [7, 7, 512],
    #                  'Resnet50': [1, 1, 2048],
    #                  'InceptionV3': [5, 5, 2048],
    #                  'Xception': [7, 7, 2048],
    #}

    net_model = Sequential()
    net_model.add(GlobalAveragePooling2D(input_shape=train_net.shape[1:]))
    #net_model.add(GlobalAveragePooling2D(input_shape=features_shape[network]))
    net_model.add(Dense(nclasses, activation='softmax'))

    print("__"+network+"__: ")
    net_model.summary()
    net_model.compile(loss='categorical_crossentropy', 
                      optimizer='rmsprop', 
                      metrics=['accuracy'])
    
    return net_model
        

def predict_file(img_path, network='Resnet50'):
    """
    Function to make prediction which label is the closest
    :param img_path: image to classify, full path
    :param network: neural network to be used
    :return: most probable label
    """

    nets = {'VGG16': bfeatures.extract_VGG16,
            'VGG19': bfeatures.extract_VGG19,
            'Resnet50': bfeatures.extract_Resnet50,
            'InceptionV3': bfeatures.extract_InceptionV3,
            'Xception': bfeatures.extract_Xception,
    }

    # clear possible pre-existing sessions. important!
    backend.clear_session()
    
    net_model = build_model(network)
    saved_weights_path = os.path.join(cfg.BASE_DIR, 'models', 
                                      'weights.best.' + network + '.hdf5')
    net_model.load_weights(saved_weights_path)
    
    # extract bottleneck features
    bottleneck_feature = nets[network](dutils.path_to_tensor(img_path))
    print("Bottleneck feature size:", bottleneck_feature.shape)
    # obtain predicted vector
    predicted_vector = net_model.predict(bottleneck_feature)
    print(predicted_vector)
    print("Sum:", np.sum(predicted_vector))
    
    labels  = dutils.labels_read(cfg.RPKIT_LabelsFile)
    print("len_labels: ", len(labels))
    for i in range(len(labels)):
        print(labels[i], " : ", predicted_vector[0][i]) 

    return mutils.format_prediction(labels, predicted_vector[0])


def predict_data(img, network='Resnet50'):
    if not isinstance(img, list):
        img = [img]
    
    filenames = []
            
    for image in img:
        f = tempfile.NamedTemporaryFile(delete=False)
        f.write(image)
        f.close()
        filenames.append(f.name)
        print("tmp file: ", f.name)

    prediction = []
    try:
        for imgfile in filenames:
            prediction.append(predict_file(imgfile, network))
    except Exception as e:
        raise e
    finally:
        for imgfile in filenames:
            os.remove(imgfile)

    return prediction


def predict_url(*args):
    message = 'Not (yet) implemented in the model (predict_url())'
    return message
        

def train(nepochs=10, network='Resnet50'):
    """
    Train network (transfer learning)
    """
    
    # check if directories for train, tests, and valid exist:
    #dutils.maybe_download_and_extract()
    
    Data_ImagesDir = os.path.join(cfg.BASE_DIR,'data', cfg.RPKIT_DataDir)
    train_files, train_targets = dutils.load_dataset(os.path.join(Data_ImagesDir,'train'))
    valid_files, valid_targets = dutils.load_dataset(os.path.join(Data_ImagesDir,'valid'))
    test_files, test_targets = dutils.load_dataset(os.path.join(Data_ImagesDir,'test'))
    
    train_net = bfeatures.load_features_set('train', network)
    valid_net = bfeatures.load_features_set('valid', network)
    test_net = bfeatures.load_features_set('test', network)
    
    print("Sizes test_files and test_targets::")    
    print(test_files.shape) #, test_targets.shape)
    print(test_files[:10])
    #print(test_targets[:10])

    #train_net, valid_net, test_net = bfeatures.load_features_all(network)
    print("Sizes of bottleneck_features (train, valid, test):")
    print(train_net.shape, valid_net.shape, test_net.shape)
    data_size = {
        'train': len(train_targets),
        'valid': len(valid_targets),
        'test': len(test_targets)
        }
    
    saved_weights_path = os.path.join(cfg.BASE_DIR, 'models', 
                                     'weights.best.' + network + '.hdf5')
    checkpointer = ModelCheckpoint(filepath=saved_weights_path, verbose=1, save_best_only=True)

    # clear possible pre-existing sessions. important!
    backend.clear_session()
 
    net_model = build_model(network)
        
    net_model.fit(train_net, train_targets, 
                  validation_data=(valid_net, valid_targets),
                  epochs=nepochs, batch_size=20, callbacks=[checkpointer], verbose=1)
    
    net_model.load_weights(saved_weights_path)
    net_predictions = [np.argmax(net_model.predict(np.expand_dims(feature, axis=0))) for feature in test_net]
    
    # report test accuracy
    test_accuracy = 100.*np.sum(np.array(net_predictions)==np.argmax(test_targets, axis=1))/float(len(net_predictions))
    print('Test accuracy: %.4f%%' % test_accuracy)
    

    return mutils.format_train(network, test_accuracy, nepochs, data_size)
