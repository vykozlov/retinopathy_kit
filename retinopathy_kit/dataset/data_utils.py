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


def rclone_copy(src_path, dest_dir, src_type='file', verbose=False):
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
            output, error = rclone_call(dest_path, dest_dir, 
                                        cmd='ls', get_output = True)
            file_size = [ elem for elem in output.split(' ') if elem.isdigit() ][0]
            print('[INFO] Copied to %s %s bytes' % (dest_path, file_size))
            dest_exist = True
            if verbose:
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
            data_zip.extractall(unzip_dir)
            data_zip.close()


# define function to load train, test, and validation datasets
def build_targets(data_type, data_labels=cfg.RPKIT_DataLabels):
    """
    Function to create train / validation / test one-hot encoded targets
    :param path: path to dataset images
    :return: numpy array containing onehot-encoded classification labels
    """
    data_dir = os.path.join(cfg.BASE_DIR, 'data', data_type)

    # load file list (full path)
    file_list = load_data_files(data_dir)
    
    # remove file extension, i.e. everything after first ".", and take basename
    files = [ os.path.basename(f).split('.', 1)[0] for f in file_list ]    
    
    df = pd.read_csv(data_labels, index_col='image')
    print(df.head(10))
    levels = []    
    for one_file in files:
        #print("%s : %i" % (one_file, df.loc[one_file]))
        levels.append(df.loc[one_file, 'level'])
    
    print("One-hot encoding check:")
    print(levels[:5])
    rpkit_targets = np_utils.to_categorical(levels, 5)
    print(rpkit_targets[:5])
    print(np.argmax(rpkit_targets, axis=1)[:5])    
    targets_file = 'RPKIT_targets_' + data_type + '.npz'
    targets_path = os.path.join(cfg.BASE_DIR, 'data', targets_file)
    
    if data_type == 'train':
        np.savez(targets_path, train=rpkit_targets)
    elif data_type == 'test':
        np.savez(targets_path, test=rpkit_targets)
    elif data_type == 'valid':
        np.savez(targets_path, valid=rpkit_targets)
    else:
        np.savez(targets_path, features=rpkit_targets)
    
    print("[INFO] Targets file shape (%s): %s" % (data_type, rpkit_targets.shape) )   
    
    return rpkit_targets

def load_data_files(data_path):
    """
    Function to load train / validation / test file names
    :param data_path: path to dataset images
    :return: numpy array containing file paths to images
    """    
    file_list = [ os.path.join(data_path, f) 
                  for f in os.listdir(data_path) 
                  if os.path.isfile(os.path.join(data_path, f)) ]
                     
    file_list.sort()

    return np.array(file_list)

def load_targets(data_type):
    """Load targets from the file
       Only one dataset, e.g. train, valid, test is loaded
    """
    
    targets_file = 'RPKIT_targets_' + data_type + '.npz'
    targets_path = os.path.join(cfg.BASE_DIR, 'data', targets_file)
    print("[INFO] Using %s" % targets_path)
    targets = np.load(targets_path)[data_type]

    return targets    

def categories_create(remote_storage=cfg.RPKIT_Storage, 
                      categories_path=cfg.RPKIT_Categories):
    """
    Function to create categories file for retinopathy.
    :return:  list of string-valued categories
    """
    categories = ['not_found' , 'mild', 'moderate', 'severe', 'proliferative']

    with open(categories_path, 'w') as listfile:
        for item in categories:
            listfile.write("%s\n" % item)

    dest_dir = remote_storage.rstrip('/') + '/data'
    print("[INFO] Upload %s to %s" % (categories_path, dest_dir))    
    rclone_copy(categories_path, dest_dir)
            
    return categories

def categories_read(categories_path=cfg.RPKIT_Categories):
    """
    Function to return categories read from the file.
    :return:  list of string-valued categories
    """
    # we expect that file already exists    
    with open(categories_path, 'r') as listfile:
        labels = [ line.rstrip('\n') for line in listfile ]        

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
