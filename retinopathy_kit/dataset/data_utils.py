# -*- coding: utf-8 -*-
import os
import zipfile
import subprocess
import numpy as np
import pandas as pd
import retinopathy_kit.config as cfg       
from keras.utils import np_utils

from keras.preprocessing import image                  
from tqdm import tqdm

### dirty trick for 'truncated images':
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
###


def rclone_call(src_path, dest_dir, cmd = 'copy', get_output=False):
    """ Function
        rclone calls
    """
    if cmd == 'copy':
        command = (['rclone', 'copy', '--progress', src_path, dest_dir]) #'--progress', 
    elif cmd == 'ls':
        command = (['rclone', 'ls', '-L', src_path])
    elif cmd == 'check':
        command = (['rclone', 'check', src_path, dest_dir])
    
    if get_output:
        result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        result = subprocess.Popen(command, stderr=subprocess.PIPE)
    output, error = result.communicate()
    return output, error


def rclone_copy(src_path, dest_dir, src_type='file'):
    """ Function for rclone call to copy data (sync?)
    :param src_path: full path to source (file or directory)
    :param dest_dir: full path to destination directory (not file!)
    :param src_type: if source is file (default) or directory
    :return: if destination was downloaded, and possible error 
    """

    error_out = None
    
    if src_type == 'file':
        src_dir = os.path.dirname(src_path)
        dest_file = src_path.split('/')[-1]
        dest_path = os.path.join(dest_dir, dest_file)
    else:
        src_dir = src_path
        dest_path =  dest_dir

    # check first if we find src_path
    output, error = rclone_call(src_path, dest_dir, cmd='ls')
    if error:
        print('[ERROR] %s (src):\n%s' % (src_path, error))
        error_out = error
        dest_exist = False
    else:
        # if src_path exists, copy it
        output, error = rclone_call(src_path, dest_dir, cmd='copy')
        if not error:
            # compare two directories, if copied file appears in output
            # as not found or not matching -> Error
            print('[INFO] File %s copied. Check if (src) and (dest) really match..' % (dest_file))
            output, error = rclone_call(src_dir, dest_dir, cmd='check')
            if 'ERROR : ' + dest_file in error:
                print('[ERROR] %s (src) and %s (dest) do not match!' % (src_path, dest_path))
                error_out = 'Copy failed: ' + src_path + ' (src) and ' + \
                             dest_path + ' (dest) do not match'
                dest_exist = False
            else:
                output, error = rclone_call(dest_path, dest_dir, 
                                            cmd='ls', get_output = True)
                file_size = [ elem for elem in output.split(' ') if elem.isdigit() ][0]
                print('[INFO] Checked: Successfully copied to %s %s bytes' % (dest_path, file_size))
                dest_exist = True
        else:
            print('[ERROR] %s (src):\n%s' % (dest_path, error))
            error_out = error
            dest_exist = False

    return dest_exist, error_out

def maybe_download_data(remote_storage = cfg.RPKIT_Storage, 
                        data_dir = '/models/bottleneck_features',
                        data_file = 'Resnet50_features_train.npz'):
    """
    Download data if it does not exist locally.
    :param data_dir: remote _and_ local dir to put data
    :param data_file: name of the file to download
    """
    # status for data if exists or not
    status = False
    error_out = None

    data_dir = data_dir.lstrip('/') #does not join if data_dir starts with '/'!
    data_dir = data_dir.rstrip('/')
    
    #check that every sub directory exists locally, if not -> create
    data_subdirs = data_dir.split('/')
    sub_dir = cfg.BASE_DIR
    for sdir in data_subdirs:
        sub_dir = os.path.join(sub_dir, sdir)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

    remote_url = remote_storage.rstrip('/') + '/' + \
                 os.path.join(data_dir, data_file)

    local_dir = os.path.join(cfg.BASE_DIR, data_dir)
    local_path = os.path.join(local_dir, data_file)
    # if data_file does not exist locally, download it
    if not os.path.exists(local_path):
        print("[INFO] Url: %s" % (remote_url))
        print("[INFO] Local path: %s" % (local_path))        
        status, error_out = rclone_copy(remote_url, local_dir)
    else:
        status = True
        error_out = None
        
    return status, error_out

def maybe_download_and_unzip(data_storage=cfg.RPKIT_Storage,
                             data_dir='/data/raw',
                             data_file='train.zip'):
    """Download and extract the zip archive.
    """
  
    # for now we assume that everything gets unzipped in ~/data directory
    unzip_dir = os.path.join(cfg.BASE_DIR, 'data')
  
    # remove last extension, should be .zip
    data_name = os.path.splitext(data_file)[0]

    # if 'data_name' is not present locally, 
    # try to download and de-archive corresponding .zip file
    if not os.path.exists(os.path.join(cfg.BASE_DIR, unzip_dir, data_name)):
        # check if .zip file present in locally
        status, _ = maybe_download_data(data_storage, data_dir, data_file)

        # if .zip is present locally, de-archive it
        file_path = os.path.join(cfg.BASE_DIR, data_dir, data_file)
        if os.path.exists(file_path):
            data_zip = zipfile.ZipFile(file_path, 'r')
            data_zip.extractall(data_dir)
            data_zip.close()


# define function to load train, test, and validation datasets
def load_dataset(data_path):
    """
    Function to load train / validation / test datasets
    :param path: path to dataset images
    :return: numpy array containing file paths to images, numpy array containing onehot-encoded classification labels
    """

    file_list = [ os.path.join(data_path, f) 
                  for f in os.listdir(data_path) 
                  if os.path.isfile(os.path.join(data_path, f)) ]
                     
    file_list.sort()
    
    # remove file extension, i.e. everything after first "."
    files = [ os.path.basename(f).split('.', 1)[0] for f in file_list ]
    
    df = pd.read_csv(cfg.RPKIT_LabelsTrain, index_col='image')
    print(df.head(10))
    print("10017_left: ", df.loc['10017_left', 'level'])

    levels = []    
    for one_file in files:
        levels.append(df.loc[one_file, 'level'])
        #print("%s : %i" % (one_file, df.loc[one_file]))
    
    print("One-hot encoding check:")
    print(levels[:5])
    targets = np_utils.to_categorical(levels, 5)
    print(targets[:5])
    print(np.argmax(targets, axis=1)[:5])
    
    return np.array(file_list), np.array(targets)

def categories_create(categories_file):
    """
    Function to create categories file for retinopathy.
    :return:  list of string-valued categories
    """
    categories = ['not_found' , 'mild', 'moderate', 'severe', 'proliferative']

    with open(categories_file, 'w') as listfile:
        for item in categories:
            listfile.write("%s\n" % item)
    return categories

def categories_read(categories_file=cfg.RPKIT_Categories):
    """
    Function to return categories read from the file.
    :return:  list of string-valued categories
    """
    
    if os.path.isfile(categories_file):
        with open(categories_file, 'r') as listfile:
            labels = [ line.rstrip('\n') for line in listfile ]
    else:
        print("[WARNING] File %s doesn't exist. Trying to create ..." % (categories_file))
        labels = categories_create(categories_file)
        dest_dir = cfg.RPKIT_Storage.rstrip('/') + '/data'
        status, _ = rclone_copy(categories_file, dest_dir)        

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
