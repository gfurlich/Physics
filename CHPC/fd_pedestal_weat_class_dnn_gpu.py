#-------------------------------------------------------------------------------------#
#
#    File :                 fd_pedestal_weat_class_dnn.py
#    Author :               Greg Furlich
#    Date Created :         2019-04-08
#
#    Purpose: To Classifiy Weather of parts into clear, cloudy, and bad given the last frame in each part in a DNN
#
#    Execution :   python fd_pedestal_weat_class_dnn.py
#
#---# Start of Script #---#

## Import Python Libraries ##

import numpy as np
import itertools
import os
import math
import psutil
import re
# import sys
# from scipy import stats
import datetime
# from ROOT import TCanvas, TGraph, TH1F, TF1, TH1D

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Pandas
import pandas as pd

# Import Keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
# from keras.utils.vis_utils import plot_model
from keras.models import model_from_json

# Import Sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Import Mathplotlib
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors

## Import Training Data File Information ##

# Parent Directories #
# parent = '/uufs/chpc.utah.edu/common/home/u0949991/'
# ml_parent =  parent + 'weat_ml/'

# CHPC Scratch Space #
# scratch = '/scratch/kingspeak/serial/u0949991/'

# CHPC Local Scratch Space #
scratch = '/scratch/local/u0949991/'

# Sub Scratch Directories #
scratch_out = scratch+'DNN_out/'
# scratch_out = scratch+'out/'

# File Paths of Data for Generators #
file = scratch + 'DNN_Vectorized_Data/{0}_ped_fluct_vectorized.npy'
labels_hdf_file = scratch + 'DNN_DF/master_br_fd_ped_db_by_part_classified.csv'

# file = scratch + 'Data/fd_ped_vect_nonpadded/{0}_ped_fluct_vectorized.npy'
# labels_hdf_file = scratch + 'DF/master_br_fd_ped_db_by_part_classified.csv'

# Out Models #
ml_weights_out = scratch_out + 'dnn_parameters.h5'
ml_model_out = scratch_out + 'dnn_model.json'

# Data Variables #
nrows, ncols = 2 * 16, 6 * 16

class_names = ['clear','cloudy','noisy']
n_classes = 3

# DNN Model Options #
nepochs = 75
model_loss = keras.losses.categorical_crossentropy
model_metric = keras.metrics.categorical_accuracy
# model_optimizer = optimizer=keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model_optimizer = keras.optimizers.Adadelta()

## Print Summary ##
model_sum = '''
== DNN Optimizer INFO ==========================

Training Epochs =\t{0}

Training Loss :\t{1}
Training Optimizer :\t{2}
Training Metrics :\t{3}

Validation Accuracy =\t{4}
Validation Cross Entropy =\t{5}

DNN Model Process Time :\t{6}

== DNN Optimizer INFO ==========================
'''

def weather_class():

    # Start Timer #
    t_start = datetime.datetime.now()

    # Check Memory #
    # print_process_mem()
    # print_mem_info()

    # Print Tensor Flow Version :
    #print(tf.__version__, pd.__version__)

    ## List Training Data Files and Load Labels DataFrame #

    # Load BR Weather info CSV to Pandas DataFrame
    df = pd.read_csv(labels_hdf_file)

    # Create Index data and Remove Missing Data #
    site = '0'
    ymdsp = df[df['part_weather_status'] != 'Missing'].apply(lambda row : 'y{0}m{1}d{2}s{3}p{4}'.format(row['run_night'].split('-')[0], row['run_night'].split('-')[1], row['run_night'].split('-')[2], site, row['part']), axis=1)
    print('Found {0} Non Missing Parts out of {1} Parts. Made {2} Indexes'.format(len(df[df['part_weather_status'] != 'Missing']), len(df), len(ymdsp)))


    # Create Weather DataFrame and remove missing info #
    weat_df = df[['run_night','part','part_weather_status']][df['part_weather_status'] != 'Missing']
    weat_df.index = ymdsp
    weat_df.index.name = 'part_id'

    files = [ file.format(index) for index, row in weat_df.iterrows()]

    ## Training and Validation Split ##
    # seed = 20181101
    # np.random.seed(seed)
    split = np.random.permutation(len(files)).tolist()
    # print(split[:10])
    split_size = int(len(files) / 3)
    val_i, train_i = split[:split_size], split[split_size:]
    train_files = [files[i] for i in train_i]
    val_files = [files[i] for i in val_i]
    train_labels = [weat_df['part_weather_status'][i] for i in train_i]
    val_labels = [weat_df['part_weather_status'][i] for i in val_i]

    # print(train_files[:10])
    # print(val[:10], train[:10], len(val), len(train))

    ## Generators ##
    print('Constructing Training and Validation Generators')
    batch_size = 64
    train_gen = partLastFrameDataGenerator(in_files = train_files, labels = train_labels, batch_size = batch_size)
    val_gen = partLastFrameDataGenerator(in_files = val_files, labels = val_labels, batch_size = batch_size)

    ## Training Data ##
    val_X, val_y = partLastFrameDataGenerator(in_files = val_files, labels = val_labels, batch_size = len(val_files), shuffle=False).__getitem__(0)
    # test = partFlattenFramesDataGenerator(in_files = val_files, labels = val_labels, batch_size = len(val_files), shuffle=False).__getfiles__(0)
    # test = partFlattenFramesDataGenerator(in_files = val_files, labels = val_labels, batch_size = len(val_files), shuffle=False).__getymdps__(0)
    # print(len(test),test[:10])

    # print_process_mem()
    # print_mem_info()

    ## DNN Model ##

    # Validation Set Shape #
    X, y = train_gen.__getitem__(0)
    X_shape, y_shape = X.shape, y.shape
    print('Input Shapes Train {0} and Label {1} '.format(X_shape, y_shape))
    print(X[-1], X[-1].shape)

    # nepochs = 125

    # Construct model #
    print('Constructing DNN Model...')

    model = keras.Sequential([

        #Input Layer Dense Layer #
        keras.layers.Flatten(input_shape=(nrows,ncols)),
        keras.layers.Dense(48, activation='relu'),
        keras.layers.Dropout(0.3),

        # Third Dense Layer #
        keras.layers.Dense(12, activation='relu'),
        keras.layers.Dropout(0.3),

        # Third Dense Layer #
        keras.layers.Dense(6, activation='relu'),

        # Classification Output Layer #
        keras.layers.Dense(n_classes, activation='softmax')

        ])

    # Compile Model #
    model.compile(loss=model_loss , optimizer=model_optimizer, metrics=[model_metric])

    # Print Model Summary #
    print(model.summary())

    # print_process_mem()
    # print_mem_info()

    # Fit Model with Test Generator #
    print('Training DNN Model...')
    history = model.fit_generator(generator=train_gen, validation_data=val_gen, epochs=nepochs, verbose=0)

    # print_process_mem()
    # print_mem_info()

    # Evaluate Model with validation Generator #
    print('Evaluating DNN Model...')
    scores = model.evaluate_generator(val_gen, verbose=0)

    print('Validation Accuracy of DNN Model : {0:.2f}% Loss:  {1:.2f}'.format( scores[1]*100,  scores[0]))

    ## Plotting model performance ##
    print('Plotting DNN Model Performance...')

    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    ## Loss Plot ##
    mpl.style.use('ggplot')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(epochs, loss, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss, 'g', label='Validation Loss')
    plt.title('DNN Training and Validation Loss')
    ax.set_ylim(bottom=0)
    ax.set_xlim(0,nepochs)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.text(0.45, 0.5, 'Preliminary', horizontalalignment='center', verticalalignment='center', color='gray', fontsize=28, fontweight='bold', rotation=30, transform=ax.transAxes)
    plt.savefig(scratch_out+'dnn_crossentropy.png', bbox_inches='tight')
    plt.clf()

    ## Plot Training Accuracy Plot ##
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(epochs, acc, 'bo', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'g', label='Validation Accuracy')
    plt.title('DNN Training and Validation Accuracy')
    ax.set_ylim(0, 1)
    ax.set_xlim(0,nepochs)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    # plt.text(0.45, 0.5, 'Preliminary', horizontalalignment='center', verticalalignment='center', color='gray', fontsize=28, fontweight='bold', rotation=30, transform=ax.transAxes)
    plt.savefig(scratch_out+'dnn_accuracy.png', bbox_inches='tight')
    plt.clf()

    ## Validation Confusion Matrix ##
    val_y_pred = model.predict(val_X)
    val_y_pred_classes = np.argmax(val_y_pred, axis=-1)
    val_y_true_classes = np.argmax(val_y, axis=-1)
    print(val_y_true_classes.shape, val_y_pred_classes.shape)
    cnf_matrix = confusion_matrix(np.array(val_y_true_classes), np.array(val_y_pred_classes))

    ## Plot and Save confusion matrix ##
    mpl.style.use('classic')
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='DNN Validation Set Normalized Confusion Matrix')
    plt.savefig(scratch_out+'dnn_val_cfm_norm.png', bbox_inches='tight')
    plt.clf()

    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False, title='DNN Validation Set Confusion Matrix')
    plt.savefig(scratch_out+'dnn_val_cfm.png', bbox_inches='tight')
    plt.clf()

    ## Save DNN Model ##
    print('Saving Trained DNN Model to {0}'.format(ml_model_out))
    print('Saving Trained DNN Weights to {0}'.format(ml_weights_out))

    # serialize model to JSON
    model_json = model.to_json()
    with open(ml_model_out, 'w') as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(ml_weights_out)

    ## Print Summary ##
    t_elapsed = datetime.datetime.now() - t_start

    print(model_sum.format(
        nepochs,
        str(model_loss),
        str(model_optimizer),
        str(model_metric),
        scores[1]*100,
        scores[0],
        t_elapsed
    ))

def plot_confusion_matrix(cm, classes, normalize=False,title='Confusion matrix', cmap=plt.cm.Reds):
    '''
    This function prints and plots the confusion matrix. Normalization can be applied by setting `normalize=True`.
    '''

    # Print Normalizaiton info :
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        cm
        # print('Confusion matrix, without normalization')

    # print(cm)

    # Create Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # cbar = plt.colorbar(cax, ticks=[ 0, 1],)
    # cbar.set_ticklabels(['0%', '100%'])

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, '{0:.2f}'.format(cm[i, j]), horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')
            # plt.text(j, i, '{0:.0f}%'.format(cm[i, j] * 100), horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')
        else :
            plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')

    # plt.text(0.33, 0.5, 'Preliminary', horizontalalignment='center', verticalalignment='center', color='gray', fontsize=28, fontweight='bold', rotation=45, transform=ax.transAxes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

## Generator for Loading Each Part's Numpy Vectorized Padded Array ##
class partLastFrameDataGenerator(keras.utils.Sequence):
    '''
    Generates data for Keras to load array from select numpy files and corresponding label stored in Pandas DataFrame
    '''

    def __init__(self, in_files, labels, batch_size=32, frame_dim=(32,96), n_classes=3, shuffle=True):
        '''
         part Data Generator Initialization
        '''

        # Generator Parameters #
        self.frame_dim = frame_dim
        self.batch_size = batch_size
        self.labels = labels
        self.n_classes = n_classes
        self.in_files = in_files
        self.shuffle = shuffle

        # Generator Functions #
        self.on_epoch_end()

    def __len__(self):
        '''
        Number of batches per training epoch
        '''

        return int(np.floor(len(self.in_files) / self.batch_size))

    def __getitem__(self, index):
        '''
        Generate one batch of data
        '''

        # Generate indexes of the batch
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        batch_files = [ self.in_files[k] for k in batch_indexes]


        # Generate Data #
        X = self.__data_generation(batch_files)

        # Generate Labels as 1 Hots Categorical #
        y = np.empty((self.batch_size), dtype=int)
        for j, k in enumerate(batch_indexes):
            y[j] = self.labels[k]

        y = keras.utils.to_categorical(y, num_classes=self.n_classes)

        return X, y

    def __getfiles__(self,index):
        '''
        Generate list of dates corresponding to batch data
        '''
        # Generate indexes of the batch
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        batch_files = [ self.in_files[k] for k in batch_indexes]

        return batch_files

    def __getymdps__(self,index):
        '''
        Generate list of dates corresponding to batch data
        '''
        # Generate indexes of the batch
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        batch_files = [ self.in_files[k] for k in batch_indexes]

        # Find YMDPS from files #
        batch_ymdps =  [re.findall('y\d+m\d+d\d+s\d+p\d+', file)[0] for file in batch_files]

        return batch_ymdps

    def on_epoch_end(self):
        '''
        Updates indexes after each epoch
        '''

        self.indexes = np.arange(len(self.in_files))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, in_files):
        '''
        Generates data containing batch_size samples
        '''

        # Data Array Initialization #
        X = np.empty((self.batch_size,) + self.frame_dim)

        # Generate Data #
        for i, file in enumerate(in_files):

            # Load All Frames from Data #
            frames = np.load(file)

            last_frame = frames[-1]

            X[i,] = last_frame

        return X

## Generator for Loading Each Part's Numpy Vectorized Padded Array ##
class partFlattenFramesDataGenerator(keras.utils.Sequence):
    '''
    Generates data for Keras to load array from select numpy files and corresponding label stored in Pandas DataFrame
    '''

    def __init__(self, in_files, labels, batch_size=32, flat_dim=(nrows*ncols), n_classes=3, shuffle=True):
        '''
         part Data Generator Initialization
        '''

        # Generator Parameters #
        self.flat_dim = flat_dim
        self.batch_size = batch_size
        self.labels = labels
        self.n_classes = n_classes
        self.in_files = in_files
        self.shuffle = shuffle

        # Generator Functions #
        self.on_epoch_end()

    def __len__(self):
        '''
        Number of batches per training epoch
        '''

        return int(np.floor(len(self.in_files) / self.batch_size))

    def __getitem__(self, index):
        '''
        Generate one batch of data
        '''

        # Generate indexes of the batch
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        batch_files = [ self.in_files[k] for k in batch_indexes]


        # Generate Data #
        X = self.__data_generation(batch_files)

        # Generate Labels as 1 Hots Categorical #
        y = np.empty((self.batch_size), dtype=int)
        for j, k in enumerate(batch_indexes):
            y[j] = self.labels[k]

        y = keras.utils.to_categorical(y, num_classes=self.n_classes)

        return X, y

    def __getfiles__(self,index):
        '''
        Generate list of dates corresponding to batch data
        '''
        # Generate indexes of the batch
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        batch_files = [ self.in_files[k] for k in batch_indexes]

        return batch_files

    def __getymdps__(self,index):
        '''
        Generate list of dates corresponding to batch data
        '''
        # Generate indexes of the batch
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        batch_files = [ self.in_files[k] for k in batch_indexes]

        # Find YMDPS from files #
        batch_ymdps =  [re.findall('y\d+m\d+d\d+s\d+p\d+', file)[0] for file in batch_files]

        return batch_ymdps

    def on_epoch_end(self):
        '''
        Updates indexes after each epoch
        '''

        self.indexes = np.arange(len(self.in_files))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, in_files):
        '''
        Generates data containing batch_size samples
        '''

        # Data Array Initialization #
        X = np.empty((self.batch_size,) + (self.flat_dim,) )

        # Generate Data #
        for i, file in enumerate(in_files):

            # Load All Frames from Data #
            frames = np.load(file)

            # Select Last Frame
            frame = frames[-1]

            # Flatten frame from 2D to 1D #
            frame = frame.flatten()

            X[i,] = frame

        return X

def bytes2humanreadable(n_bytes):
    '''
    Convert Bytes to human readable format. Based on https://github.com/giampaolo/psutil/blob/master/scripts/meminfo.py
    '''
    symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = {}
    for i, s in enumerate(symbols):
        prefix[s] = 1 << (i + 1) * 10
    for s in reversed(symbols):
        if n_bytes >= prefix[s]:
            value = float(n_bytes) / prefix[s]
            return '%.1f%s' % (value, s)
    return "%sB" % n_bytes

def print_mem_info():
    '''
    Print Memory Usage. Based on https://github.com/giampaolo/psutil/blob/master/scripts/meminfo.py
    '''
    print('MEMORY\n------')
    nt = psutil.virtual_memory()
    for name in nt._fields:
        value = getattr(nt, name)
        if name != 'percent':
            value = bytes2humanreadable(value)
        print('%-10s : %7s' % (name.capitalize(), value))

def print_process_mem():
    '''
    Print Current Process Memory Usage :
    '''
    process = psutil.Process(os.getpid())
    process_mem = bytes2humanreadable(process.memory_info().rss)
    print('Process Memory :\t{0}'.format(process_mem))

if __name__ == '__main__':
    weather_class()
