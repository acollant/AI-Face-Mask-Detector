import numpy as np
import os
import shutil
from  configs.enums import Dataset_name

from configs.utils import (
     initialize_test_dataset, 
     initialize_train_dataset, 
     remove_directory,
     create_folder,
     count_image,
     get_image
    )


from configs.constant import (
    TRAIN_PATH,
    TEST_PATH,
    BASE_PATH,
    MASK_PATH,
    DATASETS,
    NB_SPLIT_TRAINING,
    NB_SPLIT_TEST
)

for _path in [TRAIN_PATH, TEST_PATH]:
    remove_directory(_path)


for dataset_name, dataset_info in DATASETS.items():
    #_from =  BASE_PATH + dataset_name.value + '/'
    
    nb_images = dataset_info['nb_images']
    _n =  int ( nb_images * NB_SPLIT_TRAINING) #int( count_image(path,x['img_type']) * NB_SPLIT_TRAINING )
    shuffle = np.random.permutation( nb_images )
    files = dataset_info['img_list']
    
    _to = TRAIN_PATH + dataset_name.value + '/'
    create_folder(_to)
    for i in shuffle[:_n+1]:
        shutil.copy(files[i],_to)

    _to = TEST_PATH + dataset_name.value + '/'
    create_folder(_to)
    for i in shuffle[_n:]:
        shutil.copy(files[i],_to)
