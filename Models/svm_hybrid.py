
from Models import vgg_variant as var
import logging
from tensorflow.keras.regularizers import l2
import os
from tensorflow import keras 
from keras.layers import Dense, Flatten
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
# saving model histories
from keras.callbacks import CSVLogger
import tensorflow
import tensorflow as tf 
# csv_logger = CSVLogger("history/cnn_svm_history.csv", append=True)
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
        self.file_handler = logging.FileHandler(r'logs/cnn_svm_model.log')
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
        
        # freezing the last layer 
        for layer in architecture.layers[:]:
            layer.trainable = False


        x1 = Flatten()(architecture.output)
        fully_connected = Dense(units = 128, activation = 'relu')(x1)
        prediction = Dense(
            len(categories),
            kernel_regularizer = tensorflow.keras.regularizers.l2(0.01),
            activation = 'softmax')(fully_connected)
        
        self.model = Model(inputs=architecture.input, outputs=prediction)
        self.stlogger.warning('Building the model')

    def summary(self):
        self.logger.info('Getting the summary of the model')
        return self.model.summary()

    def compile(self, loss = 'squared_hinge', optimizer = 'adam', metrics = ['accuracy',var.recall_m,var.precision_m,var.f1_m]):
        self.model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])
        self.stlogger.warning('Compiling Model')
        self.logger.info(f'Model compilation with loss = {loss},optimizer = {optimizer}')

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