# -*- coding: utf-8 -*-
import os
import sys
import zipfile
import numpy as np
import pandas as pd
import retinopathy_kit.config as cfg
from sklearn.datasets import load_files       
from keras.utils import np_utils
from glob import glob
from six.moves import urllib

from keras.preprocessing import image                  
from tqdm import tqdm

### dirty trick for 'truncated images':
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
###

def maybe_download_and_extract(dataset=cfg.RPKIT_DataDir, datasetURL=cfg.RPKIT_DatasetURL):
  """Download and extract the zip archive.
     Based on tensorflow tutorials."""
  data_dir = os.path.join(cfg.BASE_DIR,'data')
  if not os.path.exists(data_dir):
      os.makedirs(data_dir)

  rawdata_dir = os.path.join(data_dir,'raw')
  if not os.path.exists(rawdata_dir):
      os.makedirs(rawdata_dir)
  

  if not os.path.exists(os.path.join(data_dir, dataset)):
      filename = datasetURL.split('/')[-1]
      filepath = os.path.join(rawdata_dir, filename)
      
      if not os.path.exists(filepath):
          def _progress(count, block_size, total_size):
              sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                  float(count * block_size) / float(total_size) * 100.0))
              sys.stdout.flush()
          filepath, _ = urllib.request.urlretrieve(datasetURL, filepath, _progress)
          print()
          statinfo = os.stat(filepath)
          print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

      dataset_zip = zipfile.ZipFile(filepath, 'r')    
      dataset_zip.extractall(data_dir)
      dataset_zip.close()


# define function to load train, test, and validation datasets
def load_dataset(data_path):
    """
    Function to load train / validation / test datasets
    :param path: path to dataset images
    :return: numpy array containing file paths to images, numpy array containing onehot-encoded classification labels
    """
    #data = load_files(path)
    #files = np.array(data['filenames'])
    #targets = np_utils.to_categorical(np.array(data['target']), 133)
    
    #diagnosis = np.arrange(5)
    file_list = [ os.path.join(data_path, f) 
                  for f in os.listdir(data_path) 
                  if os.path.isfile(os.path.join(data_path, f)) ]
                     
    file_list.sort()
    
    print(file_list[:10])
    files = [ os.path.basename(f).split('.', 1)[0] for f in file_list ]
    print("Files: ", files[:10]) 
    
    df = pd.read_csv(cfg.RPKIT_LabelsTrain)
    print(df.head(10))
    
    #targets = [ df.loc[df['image'] == f, 'level'] for f in files ]
    #print(targets[:10])
    
    df = df.loc[df['image'].isin(files)]
    #df = df.sort_values('image', axis=0)
    print(df.head(20))
    
    levels = df['level'].values
    print(levels[:20])
    targets = np_utils.to_categorical(levels, 5)
    print(targets[:20])
    
    return np.array(file_list), np.array(targets)

def labels_create(labelsFile):
    """
    Function to create labeles for retinopathy.
    Also creates .txt file with the names
    :return:  list of string-valued labels 
    """
    labels = ['not_found' , 'mild', 'moderate', 'severe', 'proliferative']

    with open(labelsFile, 'w') as listfile:
        for item in labels:
            listfile.write("%s\n" % item)
    return labels

def labels_read(labelsFile):
    """
    Function to return labels read from the file.
    :return:  list of string-valued labels
    """
    
    if os.path.isfile(labelsFile):
        with open(labelsFile, 'r') as listfile:
            labels = [ line.rstrip('\n') for line in listfile ]
    else:
        print("Warning! File ", labelsFile, " doesn't exist. Trying to create ...")
        labels = labels_create(labelsFile)

    return labels


def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
