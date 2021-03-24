import argparse
import glob
import os
import random

import numpy as np
import shutil

from utils import get_module_logger
from random import shuffle


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /mnt/data
    """
    # TODO: Implement function
    train_files = glob.glob(data_dir + '/processed/*.tfrecord')
    shuffle(train_files)

    # create the directry
    for _dir in ["train", "val", "test"]:
        dir_path = "{}/{}".format(data_dir, _dir)
        dir_path = os.path.abspath(dir_path)
        os.makedirs(dir_path, exist_ok=True)

    
   
    # Creating directory and three splits from the processed records

    train = os.path.join(data_dir, 'train')
    os.makedirs(train, exist_ok=True)
    start = 0
    end = start + int(0.8 * len(train_files))
    for file in train_files[start:end]:
        shutil.move(file, train)
    
    val = os.path.join(data_dir, 'val')
    os.makedirs(val, exist_ok=True)
    start = end
    end = start + int(0.1 * len(train_files))
    for file in train_files[start:end]:
        shutil.move(file, val)
    
    test = os.path.join(data_dir, 'test')
    os.makedirs(test, exist_ok=True)
    for file in train_files[end:]:
        shutil.move(file, test) 
    

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)
