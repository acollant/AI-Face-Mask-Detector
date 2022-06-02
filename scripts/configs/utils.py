

import shutil
import numpy as np
from glob import glob
import os




def count_image(path= "", image_type="") -> int:
    if  path == "" : return 0
    image_type = "*."+ image_type
    if image_type == "": image_type = "*.*"

    files = glob(os.path.join(path, image_type)) 
    return len(files)


def get_image(path= "", image_type="") :
    if  path == "" : return 0
    image_type = "*."+ image_type
    if image_type == "": image_type = "*.*"

    files = glob(os.path.join(path, image_type)) 
    return files




#def initialize_train_dataset():
#    train_dataset = ImageFolder(path,transform = transforms.Compose([
#        transforms.Resize((128 ,128 )),transforms.ToTensor()
#    ]))
#    return train_dataset

#def initialize_test_dataset(path):
#    test_dataset = ImageFolder(path,transforms.Compose([
#        transforms.Resize((128 ,128 )),transforms.ToTensor()
#    ]))
#    return test_dataset


def remove_directory(path):
    try:
        if  os.path.exists(os.path.dirname(path)):
            shutil.rmtree(path)
            return True
    except OSError as err:
        print(err)
        return False


def create_folder(path):
    #os.mkdir(os.path.join(path,'valid'))
    try:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
    except OSError as err:
        print(err)
