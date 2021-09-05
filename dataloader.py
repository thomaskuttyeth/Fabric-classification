
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
        self.Augmented_images = 'augmented_images' 
        self.train_data = []
        self.train_labels = [] 
        self.test_data = [] 
        self.test_labels = [] 
        self.augmented_data = [] 
        self.augmented_labels = [] 
              
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
            print('Found a test folder in the Images directory')
            print('Not moving any images from train to test') 

    
    def get_counts(self):
        train_counts = {}
        test_counts = {} 
        for category in os.listdir(self.train_path):
            train_counts[category] = len(os.listdir(self.train_path+'/'+category))
        for category in os.listdir(self.test_path):
            test_counts[category] = len(os.listdir(self.test_path+'/'+category)) 
        return train_counts, test_counts 

   
    def load_(self,folder_path,size):  
        '''looping over each subfolders and getting the image in the form of array and the corresponding labels  ''' 
        for category in range(len(self.categories)):              
            for img in os.listdir(folder_path+'/'+self.categories[category]):
                img_path = folder_path+'/'+self.categories[category]+'/'+img
                image = cv2.imread(img_path)
                try:
                    image = Image.fromarray(image,'RGB')
                except:
                    pass
                else:                    
                    image = image.resize((size,size))
                    if folder_path =='Images/train':    
                        self.train_data.append(np.array(image)/255.0)
                        self.train_labels.append(category)
                    elif folder_path =='Images/test':
                        self.test_data.append(np.array(image)/255.0) 
                        self.test_labels.append(category) 
                    elif folder_path == 'augmented_images':
                        self.augmented_data.append(np.array(image)/255.0) 
                        self.augmented_labels.append(np.array(image)/255.0)
                    else:
                        print('Give the folder path of (train,test,augmented_images)')
    
    
    