
import logging
import os
from tensorflow import keras 
from keras.layers import Dense, Flatten
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
# saving model histories
from keras.callbacks import CSVLogger
import tensorflow as tf 
from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# csv_logger= CSVLogger("history/tlmodel_history.csv", append=True)
class TLModel:
    def __init__(self, size):
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
        self.file_handler = logging.FileHandler(r'logs/tlmodels.log')
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)
        # adding stream handler
        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setFormatter(self.formatter)
        self.stream_handler.setLevel(logging.WARNING)
        self.stlogger.addHandler(self.stream_handler)

        self.stlogger.warning('Creating model.log file in log directory')

    def build(self, categories,network):
        
        image_size = [self.input_size, self.input_size]
        architecture = network(
            input_shape=image_size + [3],
            weights='imagenet', include_top=False)
        
        for layer in architecture.layers:
            layer.trainable = False



        # adding extra convolutional layers 
        conv1 = keras.layers.Conv2D(32,(3,3), 
                            activation = 'relu', 
                            padding = 'same')(architecture.output) 

        # multiple pooling part  ------> 
        
        # pooling on the right side 
        pool_max = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        
        # getting pooling on the left side 
        nega = tf.math.negative(conv1) 
        nega_pool = keras.layers.MaxPooling2D(pool_size = (2,2))(nega) 
        # taking negativ again 
        pool_range = tf.math.negative(nega_pool) 
        # gettting the range pooling 
        range_pool = tf.math.subtract(pool_max,pool_range)
        
        # concatenating pools 
        concatenated_pool1 = tf.keras.layers.Concatenate()([pool_max, range_pool]) 
        
        norm1 = keras.layers.BatchNormalization(axis = -1)(concatenated_pool1) 
        drop1 = keras.layers.Dropout(rate = 0.2)(norm1)  # regularization 



        # second additional convolutional layers 
        conv2 = keras.layers.Conv2D(32,(3,3), 
                            activation = 'relu', 
                            padding = 'same')(drop1) 
        # pool2 = keras.layers.MaxPooling2D(pool_size = (2,2))(conv2) 
        
    # multiple pooling part  ------> 
        pool_max2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        nega2 = tf.math.negative(conv2) 
        nega_pool2 = keras.layers.MaxPooling2D(pool_size = (2,2))(nega2) 
        pool_range2 = tf.math.negative(nega_pool2) 
        range_pool2 = tf.math.subtract(pool_max2,pool_range2)
        concatenated2 = tf.keras.layers.Concatenate()([pool_max2, range_pool2]) 
        
        norm2 = keras.layers.BatchNormalization(axis = -1)(concatenated2) 
        drop2 = keras.layers.Dropout(rate = 0.2)(norm2)  # regularization 

        x1 = Flatten()(drop2)
        x2 = Dense(32, activation = 'relu')(x1) 
        prediction = Dense(len(categories), activation='softmax')(x2)
        self.model = Model(inputs=architecture.input, outputs=prediction)
        self.stlogger.warning('Building the model')

    def summary(self):
        self.logger.info('Getting the summary of the model')
        return self.model.summary()

    def compile(self, loss, optimizer):
        self.model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy',recall_m,precision_m,f1_m])
        self.stlogger.warning('Compiling Model')

    # fit the model
    def fit(self, epochs, train_data, test_data):

        self.logger.info('Model Training started using fit_generator method')
        return self.model.fit_generator(
            train_data,
            validation_data=test_data,
            epochs=epochs,
            steps_per_epoch=len(train_data),
            validation_steps=len(test_data)
        )

    def predict(self, test_data):
        self.logger.info('Model prediction on the passed test data')
        return self.model.predict(test_data)

    def save(self, name_model):
        mdl = "{}.h5".format(name_model)
        self.model.save(mdl)
        self.logger.info(f'saving the final model as {name_model}.h5')

    def plot_performance(self, history):
        self.logger.info('Plotting model performance')
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        f.suptitle('CNN PERFORMANCE', fontsize=12)
        f.subplots_adjust(top=0.85, wspace=0.3)

        max_epoch = len(history.history['accuracy']) + 1
        epoch_list = list(range(1, max_epoch))
        ax1.plot(
            epoch_list, history.history['accuracy'], label='Train Accuracy')
        ax1.plot(
            epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_xticks(np.arange(1, max_epoch, 5))
        ax1.set_ylabel('ACCURACY VALUE')
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
