# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 21:29:57 2018

@author: valentin
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pkg_resources
import retinopathy_kit.config as cfg
import retinopathy_kit.dataset.data_utils as dutils
import retinopathy_kit.models.model_utils as mutils
import retinopathy_kit.features.build_features as bfeatures
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import backend
from keras import optimizers
from keras.metrics import top_k_categorical_accuracy

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from six.moves import cPickle as pickle

#from sklearn.ensemble import RandomForestClassifier

input_shape = [16, 16, 8]

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
        

def prepare_data(network='Resnet50'):
    """ Function to prepare data
    """
    # check if categories file exists locally, if not -> download,
    # if not downloaded -> dutils.categories_create()
    status_categories, _ = dutils.maybe_download_data(data_dir='/data', 
                                                      data_file=cfg.RPKIT_Categories)
    if not status_categories:
        dutils.categories_create()

    # check if trainLabels.csv file exists locally, if not -> download,
    status_labels_train, _ = dutils.maybe_download_data(data_dir='/data', 
                                                        data_file=cfg.RPKIT_LabelsTrain)
                                                        
    # check if bottleneck_features file fexists locally
    # if not -> download it, if not downloaded -> try to build
    # train
    bfeatures.check_features_set('train', network)
    # valid
    bfeatures.check_features_set('valid', network)
    # test                                                         
    bfeatures.check_features_set('test', network)


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
    #input_shape = train_net.shape[1:]  ## 16, 16, 8
    ###net_model.add(Conv2D(256, (3,3), padding="same", input_shape=train_net.shape[1:], activation="relu"))
    #+net_model.add(Conv2D(128, (3,3), padding="same", input_shape=train_net.shape[1:], activation="relu"))
    net_model.add(Conv2D(256, (3,3), padding="same", input_shape=input_shape, activation="relu"))
    net_model.add(Dropout(0.25))
    #-net_model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
    #-net_model.add(Conv2D(256, (5, 5), input_shape=train_net.shape[1:], padding='same'))
    #-net_model.add(Activation('relu'))
    #-net_model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
    #-net_model.add(Conv2D(384, (3, 3), padding='same'))
    #-net_model.add(Activation('relu'))
    # skip two following lines. without them the model learns a bit faster and better (?)
    ##-inmodel.add(Conv2D(256, (3, 3)))
    ##-inmodel.add(Activation('relu'))
    #-net_model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

    net_model.add(GlobalAveragePooling2D())
    net_model.add(Dense(128, activation='relu'))   #64
    net_model.add(Dense(128, activation='relu'))   #64
    net_model.add(Dense(nclasses, activation='softmax'))

    print("__"+network+"__: ")
    def top_2_accuracy(in_gt, in_pred):
        return top_k_categorical_accuracy(in_gt, in_pred, k=2)    
    #opt = optimizers.RMSprop(lr=0.0002, rho=0.9, epsilon=0.1, decay=0.001)
    opt = optimizers.Adagrad(lr=0.002)
    net_model.compile(loss='categorical_crossentropy',
                      optimizer='adam',    #opt, 'rmsprop', 'adagrad', 'adam'
                      metrics=['categorical_accuracy', top_2_accuracy])

    net_model.summary()
    
    return net_model

def predict_cnn(img_path, network='Resnet50'):
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
    bottleneck_feature = bottleneck_feature.reshape([bottleneck_feature.shape[0]] + input_shape) ##exp ## 16,16,8
    print("Bottleneck feature size:", bottleneck_feature.shape)
    # obtain predicted vector
    predicted_vector = net_model.predict(bottleneck_feature)
    print("=> %s : index of max: %d" % 
          (os.path.basename(img_path), np.argmax(predicted_vector[0])))
    print(predicted_vector)
    print("Sum:", np.sum(predicted_vector))
    
    return predicted_vector
    
def predict_logreg(img_path, network='Resnet50'):
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
      
    saved_model_path = os.path.join(cfg.BASE_DIR, 'models', 
                                      'logreg.best.' + network + '.pkl')

    with open(saved_model_path, 'rb') as file:  
        net_model = pickle.load(file)
    
    # extract bottleneck features
    bottleneck_feature = nets[network](dutils.path_to_tensor(img_path))
    bottleneck_feature = bottleneck_feature.reshape((bottleneck_feature.shape[0], 2048))    
    print("Bottleneck feature size:", bottleneck_feature.shape)
    # obtain predicted vector
    predicted_vector = net_model.predict_proba(bottleneck_feature)
    print("=> %s : index of max: %d" % 
          (os.path.basename(img_path), np.argmax(predicted_vector[0])))
    print(predicted_vector)
    print("Sum:", np.sum(predicted_vector))
    
    return predicted_vector    
        
def predict_file(img_path, network='Resnet50', model='cnn'):
    """
    Function to make prediction which label is the closest
    :param img_path: image to classify, full path
    :param network: neural network to be used
    :return: most probable label
    """

    categories  = dutils.categories_read()
    
    # obtain predicted vector
    if model == 'cnn':
        predicted_vector = predict_cnn(img_path, network)
    elif model == 'logreg':
        predicted_vector = predict_logreg(img_path, network)
      
    print("len_categories: ", len(categories))
    for i in range(len(categories)):
        print(categories[i], " : ", predicted_vector[0][i])
    
    return mutils.format_prediction(categories, predicted_vector[0])


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
        

def predict_kaggle(test_path, sample_file, network='Resnet50'):
    '''
    Function to produce .csv output for Kaggle score
    :param test_path: path to unseen test data
    :param sample_file: sampleSubmission file to take list of necessary files
    '''
  
    # store output file where sample_file is located
    output_path = os.path.dirname(sample_file)
    sample_filename = sample_file.split('/')[-1]
    output_file_pfx = sample_filename.split('.', 1)[0]

    # define sub-function for inference
    def predict_batch(network, img_paths):
        
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
        bottleneck_features = nets[network](dutils.paths_to_tensor(img_paths))
        bottleneck_features = bottleneck_features.reshape([bottleneck_features.shape[0]] + input_shape)  ## 16,16,8
        predictions_batch = [np.argmax(net_model.predict(np.expand_dims(feature, axis=0))) for feature in bottleneck_features]
        
        return predictions_batch

    # read sample_file in Pandas Dataframe. take 'image' column as index
    df = pd.read_csv(sample_file, index_col='image')
    #print(df.head(10))
    imgs = []
    imgs_batch = [] 
    imgs_batch_paths = []
    predictions = []
    idx_int = 0
    batch_size = 5000
    for idx, row in df.iterrows():
        #print(idx, df['level'][idx])
        #print("=> ", idx, df.index[idx_int], idx_int)
        imgs.append(idx)
        imgs_batch.append(idx)
        imgs_batch_paths.append(os.path.join(test_path, idx + '.jpeg'))
        # do prediction on batch_size of samples (doing 1-by-1 is tooo slow!)
        if idx_int%batch_size == 0 and idx_int > 0:
            print("=> Predicting %i iterations %i, %s" %(batch_size, idx_int, idx))
            print("=> N images = ", len(imgs_batch))
            predictions_batch = predict_batch(network, imgs_batch_paths)
            ##predictions_batch = [1]*len(imgs_batch) ##for tests
            predictions.extend(predictions_batch)
            df_batch = pd.DataFrame({"image" : imgs_batch, "level" : predictions_batch})
            output_batch = os.path.join(output_path, output_file_pfx + str(idx_int) + '.csv')
            # store intemediate results
            df_batch.to_csv(output_batch, index=False)
            # reset 'batch' arrays
            imgs_batch = []
            imgs_batch_paths = []
            
        # do prediction on remaining files if full sample is not integer of batch_size           
        if idx == df.index[-1] and idx_int%batch_size > 0:
            print("=> Predicting last iteration ", idx_int, idx)
            predictions_batch = predict_batch(network, imgs_batch_paths)
            ##predictions_batch = [0]*len(imgs_batch)  ##for tests
            predictions.extend(predictions_batch)
            df_batch = pd.DataFrame({"image" : imgs_batch, "level" : predictions_batch})
            output_batch = os.path.join(output_path, output_file_pfx + str(idx_int) + '.csv')
            df_batch.to_csv(output_batch, index=False)
        
        idx_int += 1
            
    # create final output file 
    df_all = pd.DataFrame({"image" : imgs, "level" : predictions})
    output_all = os.path.join(output_path, output_file_pfx + '.csv')
    df_all.to_csv(output_all, index=False)
    

def train_cnn(nepochs=10, network='Resnet50'):
    """
    Train network (transfer learning)
    """
    
    # check if directories for train, tests, and valid exist:
    #dutils.maybe_download_and_extract()
    
    Data_ImagesDir = os.path.join(cfg.BASE_DIR,'data')
    train_files, train_targets = dutils.load_dataset(os.path.join(Data_ImagesDir,'train'))
    valid_files, valid_targets = dutils.load_dataset(os.path.join(Data_ImagesDir,'valid'))
    test_files, test_targets = dutils.load_dataset(os.path.join(Data_ImagesDir,'test'))
    
    train_net = bfeatures.load_features_set('train', network)
    valid_net = bfeatures.load_features_set('valid', network)
    test_net = bfeatures.load_features_set('test', network)
    
    train_net = train_net.reshape([train_net.shape[0]] + input_shape)  ##16, 16, 8
    valid_net = valid_net.reshape([valid_net.shape[0]] + input_shape)
    test_net = test_net.reshape([test_net.shape[0]] + input_shape)

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
    
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', 
                                       factor=0.8, patience=3, 
                                       verbose=1, mode='auto', 
                                       epsilon=0.0001, 
                                       cooldown=5, 
                                       min_lr=0.0001)
    early = EarlyStopping(monitor="val_loss", 
                          mode="min", 
                          patience=12) # probably needs to be more patient, but kaggle time is limited

    callbacks_list = [checkpointer, early, reduceLROnPlat]    
        
    net_model.fit(train_net, train_targets, 
                  validation_data=(valid_net, valid_targets),
                  epochs=nepochs, batch_size=16, callbacks=callbacks_list, verbose=1)
    
    net_model.load_weights(saved_weights_path)

    #test_net = test_net.reshape([test_net.shape[0]] + input_shape)  
    net_predictions = [np.argmax(net_model.predict(np.expand_dims(feature, axis=0))) for feature in test_net]
    
    # report test accuracy
    test_accuracy = 100.*np.sum(np.array(net_predictions)==np.argmax(test_targets, axis=1))/float(len(net_predictions))
    print('Test accuracy: %.4f%%' % test_accuracy)

    # generate a classification report for the model
    print(classification_report(np.argmax(test_targets, axis=1), net_predictions))
    # compute the raw accuracy with extra precision
    acc = accuracy_score(np.argmax(test_targets, axis=1), net_predictions)
    print("[INFO] score: {}".format(acc))    

    dest_dir = cfg.RPKIT_Storage.rstrip('/') + '/models'
    dutils.rclone_copy(saved_weights_path, dest_dir)

    return mutils.format_train(network, test_accuracy, nepochs, data_size)

def train_logreg(network='Resnet50'):
    '''
    Train non-neural network classifier, e.g. Logistic Regression or Random Forests 
    '''
    Data_ImagesDir = os.path.join(cfg.BASE_DIR,'data')
    train_files, train_targets = dutils.load_dataset(os.path.join(Data_ImagesDir,'train'))
    valid_files, valid_targets = dutils.load_dataset(os.path.join(Data_ImagesDir,'valid'))
    test_files, test_targets = dutils.load_dataset(os.path.join(Data_ImagesDir,'test'))

    # convert one-hot encoded targets back to 'index numbers'    
    train_targets = np.argmax(train_targets, axis=1)
    valid_targets = np.argmax(valid_targets, axis=1)
    test_targets = np.argmax(test_targets, axis=1)    
    
    train_net = bfeatures.load_features_set('train', network)
    valid_net = bfeatures.load_features_set('valid', network)
    test_net = bfeatures.load_features_set('test', network)
    
    # reshape the features so that each image is represented by
    # a flattened feature vector of the `MaxPooling2D` outputs
    # taken from https://github.com/jrosebr1/microsoft-dsvm/blob/master/pyimagesearch-22-minutes-to-2nd-place.ipynb
    newshape2 = train_net.shape[1]*train_net.shape[2]*train_net.shape[3]
    train_net = train_net.reshape((train_net.shape[0], newshape2))
    valid_net = valid_net.reshape((valid_net.shape[0], newshape2))
    test_net = test_net.reshape((test_net.shape[0], newshape2))
    
    #train_net, valid_net, test_net = bfeatures.load_features_all(network)
    print("Sizes of bottleneck_features (train, valid, test):")
    print(train_net.shape, valid_net.shape, test_net.shape)
    data_size = {
        'train': len(train_targets),
        'valid': len(valid_targets),
        'test': len(test_targets)
        }
                                     
    print("[INFO] tuning hyperparameters...")
    params = {"C": [0.001, 0.005, 0.01, 0.02, 0.1]}
    clf = GridSearchCV(LogisticRegression(solver='lbfgs'), params, cv=5, n_jobs=-1)
    #clf = RandomForestClassifier(n_estimators=256, oob_score=True, random_state=0)    
    clf.fit(train_net, train_targets)
    print("[INFO] best hyperparameters: {}".format(clf.best_params_))
    #print("[INFO] best hyperparameters: {}".format(clf.oob_score_))
       
    # generate a classification report for the model
    print("[INFO] evaluating...")
    preds = clf.predict(test_net)
    print(classification_report(test_targets, preds))

    # compute the raw accuracy with extra precision
    acc = accuracy_score(test_targets, preds)
    print("[INFO] score: {}".format(acc))

    saved_model_path = os.path.join(cfg.BASE_DIR, 'models', 
                                     'logreg.best.' + network + '.pkl')

    print("[INFO] storing model in %s ..." % (saved_model_path))
    with open(saved_model_path, 'wb') as file:  
        pickle.dump(clf.best_estimator_, file)