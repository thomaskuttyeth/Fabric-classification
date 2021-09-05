from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16

import numpy as np
import random
random.seed(5)
from  tensorflow.keras.utils import to_categorical
import logging
import os 
class TLModel:
    def __init__(self, size):
        ''' size represents the input data size) '''
        self.model = None
        self.input_size = size
        
        # logging set up 
        if 'logs' not in os.listdir():
            os.mkdir('logs')
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.stlogger = logging.getLogger(__name__)
        self.stlogger.setLevel(logging.WARNING)
        self.formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
        self.file_handler = logging.FileHandler('logs\model.log')
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)
        # adding streamhandler
        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setFormatter(self.formatter)
        self.stream_handler.setLevel(logging.WARNING)
        self.stlogger.addHandler(self.stream_handler)


    def build(self,categories):
        IMAGE_SIZE =[self.input_size, self.input_size]
        vgg = VGG16(
            input_shape = IMAGE_SIZE+[3],
            weights = 'imagenet', include_top = False)
        for layer in vgg.layers:
            layer.trainable = False
        x = Flatten()(vgg.output)
        prediction = Dense(len(categories), activation='softmax')(x)
        self.model = Model(inputs = vgg.input, outputs = prediction)
        self.stlogger.warn('done with model building')

    def summary(self):
        return self.model.summary()

    def compile(self, loss, optimizer, metrics):
        self.model.compile(loss = loss, optimizer = optimizer, metrics = [metrics])
        self.stlogger.warn('model compiled') 
# fit the model
    def fit(self, epochs,train_data,test_data ):
        return self.model.fit_generator(
            train_data,
            validation_data=test_data,
            epochs=epochs,
            steps_per_epoch=len(train_data),
            validation_steps=len(test_data))

    def test(self, test_data):
        test = np.array(test_data)
        test_final = []
        for img in test:
            try:
                img.resize(1,self.input_size, self.input_size,3)
            except Exception:
                print(Exception)
            else:
                test_final.append(img)
            self.logger.info('appending image to the final dataframe')
        predictions = []
        for img in test_final:
            predictions.append(self.predict(img))
            self.logger.info(f'Model output = {self.predict(img)}')

    def save(self, name_model):
        mdl = "{}.h5".format(name_model)
        self.model.save(mdl)
        self.logger.info(f'saving the final model as {name_model}.h5')
        
    def plot_performance(self, history):
        '''history = the model history'''
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        f.suptitle('CNN PERFORMANCE', fontsize=12)
        f.subplots_adjust(top=0.85, wspace=0.3)

        max_epoch = len(history.history['accuracy'])+1
        epoch_list = list(range(1, max_epoch))
        ax1.plot(
            epoch_list, history.history['accuracy'], label='Train Accuracy')
        ax1.plot(
            epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_xticks(np.arange(1, max_epoch, 5))
        ax1.set_ylabel('ACCCURACY VALUE')
        ax1.set_xlabel('EPOCH')
        ax1.set_title('ACCURACY')
        ax1.legend(loc='best')

        ax2.plot(epoch_list, history.history['loss'], label='Train loss')
        ax2.plot(
            epoch_list, history.history['val_loss'], label='Validation loss')
        ax2.set_xticks(np.arange(1, max_epoch, 5))
        ax2.set_ylabel('LOSS VALUE')
        ax2.set_xlabel('EPOCH')
        ax2.set_title('LOSS')
        ax1.legend(loc='best')