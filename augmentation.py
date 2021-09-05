
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from numpy.lib.type_check import _imag_dispatcher
import tensorflow as tf
import os 
import shutil

class ImageAugmentation:
    def __init__(self):
        
        '''
        Warning : Do augmentation after splitting the data into train and test folder.
        Augmentation should happen for all the images in the training subfolder under image directory.  
        parameters 
        ====================
        dir_name = name of the directory for storing augmented images 
        image_dir = the main image directory which contains train and test folder'''
        
        if 'augmented_images' in os.listdir():
            shutil.rmtree('augmented_images')
        
        self.train_path = 'Images/train'
        
        if 'augmented_images' not in os.listdir():
            os.mkdir('augmented_images') 
        
        
        # setting the subdirectories 
        for folder in os.listdir(self.train_path):
            if folder not in os.listdir('augmented_images'):
                os.mkdir('augmented_images'+'/'+folder) 
            else:
                pass
      
      # using keras to create and instance of imagedatagenerator 
        self.datagenerator  = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.1,
            horizontal_flip =True,
            fill_mode='nearest'
            )
     
    def generate_images(self,n):
        for folder in range(len(os.listdir(self.train_path))):
            for im in os.listdir(self.train_path+'/'+os.listdir(self.train_path)[folder]):
                img_path = self.train_path+'/'+os.listdir(self.train_path)[folder]+'/'+im
                image = tf.keras.preprocessing.image.load_img(img_path)
                x = img_to_array(image) 
                x = x.reshape((1,) + x.shape)
                i = 0
                for batch in self.datagenerator.flow(
                  x, batch_size = 1,
                  save_to_dir = 'augmented_images'+'/'+os.listdir(self.train_path)[folder],
                  save_prefix = 'aug_img',
                  save_format = 'jpg'):
                    i +=2
                    if i>n:
                        break
