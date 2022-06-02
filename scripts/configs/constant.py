#train and test data directory
from configs.enums import Dataset_name
from configs.utils import count_image, get_image
from torchvision import transforms 

TRAIN_PATH = "../data/masks/train/"
TEST_PATH  = "../data/masks/test/"
MASK_PATH = "../data/masks/"
BASE_PATH = "../data/base/"

NB_SPLIT_TRAINING = 0.8
NB_SPLIT_TEST = 0.2

"""
- transforms.Compose: Ro prepare a dataset from such a structure, PyTorch provides ImageFolder class 
which makes the task easy for us to prepare the dataset. We simply have to pass 
the directory of our data to it and it provides the dataset which we can use to train the model.
https://medium.com/thecyphy/train-cnn-model-with-pytorch-21dafb918f48


- transforms.Normalize: normalization helps get data within a range and reduces the skewness which helps learn faster and better. 
Normalization can also tackle the diminishing and exploding gradients problems.
https://pytorch.org/vision/stable/models.html
"""
SIMPLE_TRANSFORM = transforms.Compose([transforms.Resize((128,128))
                                       ,transforms.ToTensor()
                                       ,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


DATASETS = {
    Dataset_name.Cloth: {  
        "nb_images" : count_image( BASE_PATH + Dataset_name.Cloth + '/', 'jpeg'),
        "img_list" : get_image( BASE_PATH + Dataset_name.Cloth + '/', 'jpeg'),
        "img_type" : "jpeg"
    },
    Dataset_name.NoMask: {
        "nb_images" : count_image(BASE_PATH + Dataset_name.NoMask + '/', 'png'),
        "img_list" : get_image( BASE_PATH + Dataset_name.NoMask + '/', 'png'),
        "img_type": 'png'
    },
    Dataset_name.N95: {  
        "nb_images" : count_image(BASE_PATH + Dataset_name.N95 + '/', 'jpg'),
        "img_list" : get_image( BASE_PATH + Dataset_name.N95 + '/', 'jpg'),
        "img_type" : 'jpg',
    },
    Dataset_name.Surgical: {
        "nb_images" : count_image(BASE_PATH + Dataset_name.Surgical + '/', 'jpg'),
        "img_list" :  get_image( BASE_PATH + Dataset_name.Surgical + '/', 'jpg'),
        "img_type" : 'jpg',
    },
    Dataset_name.Improper: {
        "nb_images" : count_image(BASE_PATH + Dataset_name.Improper + '/', 'jpg'),
        "img_list" : get_image( BASE_PATH + Dataset_name.Improper + '/', 'jpg'),
        "img_type" : 'jpg',
    },
}