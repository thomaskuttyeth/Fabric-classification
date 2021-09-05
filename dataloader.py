
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
import logging 
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
        
        # logging 
                # logging set up 
        if 'logs' not in os.listdir():
            os.mkdir('logs')
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.stlogger = logging.getLogger(__name__)
        self.stlogger.setLevel(logging.INFO)
        self.formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
        self.file_handler = logging.FileHandler('logs\dataloader.log')
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)
        # adding streamhandler
        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setFormatter(self.formatter)
        self.stream_handler.setLevel(logging.INFO)
        self.stlogger.addHandler(self.stream_handler)
        
        self.stlogger.info('Created new dataloader.log file in log directory') 
        
        # creating instances of image data generator for loding training and testing images 
        self.train_datagen = ImageDataGenerator(
                rescale = 1./255,
                shear_range = 0.2,
                zoom_range = 0.2,
                horizontal_flip = True)
        self.logger.info('created ImageDataGenerator instance for train data generator') 

        self.test_datagen = ImageDataGenerator(rescale = 1./255)
        self.logger.info('Created ImageDataGenerator instance for test data genrator')
        
        # loading the training set 
        self.training_set = self.train_datagen.flow_from_directory(
            'Images/train',
            target_size = (224, 224),
            batch_size = 16,
            class_mode = 'categorical')
        self.stlogger.info('loading training set') 

        # loading the testing set 
        self.test_set = self.test_datagen.flow_from_directory(
            'Images/test',
            target_size = (224, 224),
            batch_size = 16,
            class_mode = 'categorical') 
        self.stlogger.info('loading the testing set') 
              
    def make_test(self,ratio): 
        if 'test' not in os.listdir(self.Images): 
            os.mkdir('Images/test')
            self.stlogger.info('Created Test folder in the image directory') 
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
                    self.logger.info(f'{img} moved from train to test') 

        else:
            self.stlogger.info('Test folder is already exists in the Images directory') 

    
    def get_counts(self):
        train_counts = {}
        test_counts = {} 
        for category in os.listdir(self.train_path):
            train_counts[category] = len(os.listdir(self.train_path+'/'+category))
        for category in os.listdir(self.test_path):
            test_counts[category] = len(os.listdir(self.test_path+'/'+category))
        self.logger.info('Getting the counts of each image classes in train and test') 
        return train_counts, test_counts 

   
    