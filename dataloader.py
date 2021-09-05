
# importing modules 
import os 
import shutil 
from random import shuffle 
from glob import glob 
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf 
from keras.preprocessing.image import ImageDataGenerator, img_to_array 

# creating dataloader class for loading and preprocess data 
class Dataloader:
    '''
    dataloader class helps to load images from directories and preprocess them.
    ========================
    parameters
    =========================
    img_dir = pass the image directory path which must contain train directory
    '''
    def __init__(self): 
        self.categories = [ i for i in os.listdir('Images/train')]
        self.train_path = 'Images/train' 
        self.test_path = 'Images/test' 
        self.Images = 'Images'
        self.train_datagen = ImageDataGenerator(
                rescale = 1./255,
                shear_range = 0.2,
                zoom_range = 0.2,
                horizontal_flip = True)

        self.test_datagen = ImageDataGenerator(rescale = 1./255)

        self.training_set = self.train_datagen.flow_from_directory(
            'Images/train',
            target_size = (224, 224),
            batch_size = 16,
            class_mode = 'categorical')

        self.test_set = self.test_datagen.flow_from_directory(
            'Images/test',
            target_size = (224, 224),
            batch_size = 16,
            class_mode = 'categorical') 
    
              
    def make_test(self,ratio): 
        if 'test' not in os.listdir(self.Images): 
            os.mkdir('Images/test')
            for i in self.categories:     
                if i not in os.listdir(self.test_path):
                    os.mkdir(self.test_path+'/'+i)

            # moving n random images from train subdirectory to test directory 
            for subfolder in os.listdir(self.train_path):
                images =  os.listdir(self.train_path+"/"+subfolder)
                shuffle(images) 
                # taking the first n images wrt ratio parameter 
                for img in images[0:int(len(images)*ratio)]:
                    shutil.move(
                        os.path.join(self.train_path+'/'+subfolder, img),
                        self.test_path+'/'+subfolder)

        else:
            print('Test folder is already exists in the Images directory') 

    
    def get_counts(self):
        train_counts = {}
        test_counts = {} 
        for category in os.listdir(self.train_path):
            train_counts[category] = len(os.listdir(self.train_path+'/'+category))
        for category in os.listdir(self.test_path):
            test_counts[category] = len(os.listdir(self.test_path+'/'+category)) 
        return train_counts, test_counts 

   
    