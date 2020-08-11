#-------------------------------------------------------------------------------------#
#
#    File :                 fd_pedestal_weat_classify_all_rcnn.py
#    Author :               Greg Furlich
#    Date Created :         2019-10-10
#
#    Purpose:               Read in Model for RCNN and then calssify all the pedestal data.
#
#    Execution :   python fd_pedestal_weat_classify_all_rcnn.py
#
#---# Start of Script #---#

## Import Python Libraries ##

import numpy as np
import itertools
import os
import math
import psutil
import re
import time, datetime

# import sys
# from scipy import stats
# import datetime
# from ROOT import TCanvas, TGraph, TH1F, TF1, TH1D

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import device_lib

# Pandas
import pandas as pd

# Import Keras
from tensorflow.keras.models import Sequential
# from tensorflow.keras.utils import np_utils
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import InputLayer
# from keras.wrappers.scikit_learn import KerasClassifier
# from keras.utils.vis_utils import plot_model
from tensorflow.keras.models import model_from_json

# Import Sklearn
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import RepeatedStratifiedKFold

# Import Mathplotlib
# import matplotlib as mpl
# mpl.use('agg')
# import matplotlib.pyplot as plt
# import matplotlib.colors as colors

## Import Training Data File Information ##

# Parent Directories #
# parent = '/uufs/chpc.utah.edu/common/home/u0949991/'
# ml_parent =  parent + 'weat_ml/'

# CHPC Scratch Space #
# scratch = '/scratch/kingspeak/serial/u0949991/'

# CHPC Local Scratch Space #
scratch = '/scratch/local/u0949991/'

# Sub Scratch Directories #
scratch_out = scratch+'RCNN_out/'

# File Paths of Data for Generators #
file = scratch + 'RCNN_Data/{0}_ped_fluct_vectorized.npy'
# all_br_data_to_classify_h5 = scratch + 'RCNN_DF/master_fd_ped_db_by_part.h5'
# in_br_ped_df = 'master_br_fd_ped_db_by_part'
all_br_data_to_classify_csv = scratch + 'RCNN_DF/master_br_fd_ped_db_by_part.csv'

# Out Models #
ml_weights_in = scratch + 'RCNN_DF/rcnn_parameters.h5'
ml_model_in = scratch + 'RCNN_DF/rcnn_model.json'

# Data Variables #
nrows, ncols = 2 * 16, 6 * 16
t_dim = 216

class_names = ['clear','cloudy','noisy']
n_classes = 3

# RCNN Model Options #
# nepochs = 50
model_loss = tf.keras.losses.categorical_crossentropy
model_metric = tf.keras.metrics.categorical_accuracy
# model_optimizer = keras.optimizers.SGD()
# model_optimizer = keras.optimizers.RMSprop()
# model_optimizer = keras.optimizers.Adagrad()
# model_optimizer = keras.optimizers.Adadelta()
# model_optimizer = keras.optimizers.Adam()
# model_optimizer = keras.optimizers.Adamax()
# model_optimizer = keras.optimizers.Nadam()

# Adagrad seems to be the most stable of Optimizers #
# model_optimizer = keras.optimizers.Adagrad(lr=0.0085, epsilon=None, decay=0)
model_optimizer = tf.keras.optimizers.Adagrad(lr=0.0085)

## Print Summary ##
# model_sum = '''
# == RCNN Optimizer INFO ==========================
#
# Training Epochs =\t{0}
#
# Training Loss :\t{1}
# Training Optimizer :\t{2}
# Training Metrics :\t{3}
#
# Validation Accuracy =\t{4}
# Validation Cross Entropy =\t{5}
#
# RCNN Model Process Time :\t{6}
# RCNN Model Process Date :\t{7}
#
# == RCNN Optimizer INFO ==========================
# '''

def classify_pedestal_weather():

    # Start Timer #
    t_start = datetime.datetime.now()

    # print module versions #
    print('TensorFlow Version: {0}'.format(tf.__version__))
    print('Keras Version: {0}'.format(keras.__version__))

    # Check Memory #
    print_process_mem()
    print_mem_info()

    # Print GPU Info #
    print(device_lib.list_local_devices())

    # Print Tensor Flow Version :
    #print(tf.__version__, pd.__version__)

    ## List Training Data Files and Load Labels DataFrame #

    # Load BR Weather info CSV to Pandas DataFrame
    # store = pd.HDFStore(all_br_data_to_classify_h5)
    # print(store.info())
    # ped_df = store[in_br_ped_df]
    ped_df = pd.read_csv(all_br_data_to_classify_csv)

    # Create Index data #
    site = '0'
    ymdsp = ped_df.apply(lambda row : 'y{0}m{1}d{2}s{3}p{4}'.format(row['run_night'].split('-')[0], row['run_night'].split('-')[1], row['run_night'].split(' ')[0].split('-')[2], site, row['part']), axis=1)
    # print(ymdsp.head())
    print('Found {0} BR FD Pedestal Parts'.format(len(ymdsp)))

    # Check File Animation Exists #
    br_files = [ file.format(ymdsp) for ymdsp in ymdsp]
    # print(br_files[0:10])

    # ped_df['part_status']
    status = []
    for files in br_files:

        # Check if File Exists #
        if os.path.isfile(files):

            # Check if has a File has Array #
            if np.load(files).shape[0] > 0 :
                status.append('Exists')

                # No Array is in File #
            else :
                status.append('No Array')

        # File doesn't Exist #
        else :
            status.append('No File')

    ped_df['part_status'] =  status

    print(ped_df['part_status'].value_counts())

    # Create Weather DataFrame and remove missing info #
    weat_df = ped_df[['run_night','part','part_weather_status']][ped_df['part_status'] == 'Exists']
    weat_df.index = ymdsp[ped_df['part_status'] == 'Exists']
    weat_df.index.name = 'part_id'
    print('Classifying {0} BR FD Pedestal Parts'.format(len(weat_df)))

    classify_files = [ file.format(index) for index, row in weat_df.iterrows()]
    classify_labels = weat_df['part_weather_status']

    # print(classify_files[:10])

    ## Generators ##
    print('Constructing Classifying Generators @ {0}...'.format(datetime.datetime.now() ) )
    classify_gen = partAllFramesDataGenerator(in_files = classify_files, labels = classify_labels, batch_size=16, shuffle=False)
    print(classify_gen, tf.keras.utils.Sequence)

    ## Load all data into an array ##
    # classify_X, classify_y = partAllFramesDataGenerator(in_files = classify_files, labels = classify_labels, batch_size = len(classify_files), shuffle=False).__getitem__(0)

    print( len(classify_gen))

    print_process_mem()
    print_mem_info()

    ## RCNN Model ##

    # Load model and weights #
    print('Loading RCNN Model and Weights @ {0}...'.format(datetime.datetime.now() ) )

    rcnn_json_file = open(ml_model_in, 'r')
    loaded_rcnn_model_json = rcnn_json_file.read()
    rcnn_json_file.close()
    loaded_rcnn_model = model_from_json(loaded_rcnn_model_json)
    loaded_rcnn_model.load_weights(ml_weights_in)

    print(loaded_rcnn_model.summary())

    # Compile Model #
    print('Compile RCNN Model @ {0}...'.format(datetime.datetime.now()))
    loaded_rcnn_model.compile(loss=model_loss , optimizer='adagrad', metrics=[model_metric])

    # Print Model Summary #
    # print(loaded_rcnn_model.summary())

    print_process_mem()
    print_mem_info()

    # Predict Weather Classes #
    print('Predict BR Pedestal Weather Classes with RCNN Model @ {0}...'.format(datetime.datetime.now() ) )
    # pedestal_pred = loaded_rcnn_model.predict(classify_X)
    pedestal_pred = loaded_rcnn_model.predict_generator(classify_gen)
    pedestal_pred_classes = np.argmax(pedestal_pred, axis=-1)

    print('Saving BR Pedestal Weather Classes predicted by RCNN Model @ {0}...'.format(datetime.datetime.now() ) )

    # Save Weather Classes #
    test = pd.DataFrame()
    test['weather'] = pedestal_pred_classes
    print(test.info(verbose=True))
    print(test.head())
    test['weather_class'] = test.apply(lambda row : int2weather_class(int(row)))

    print('Predicted {0} Parts Weather'.format(len(test)))
    print(test['weather'].value_counts())
    print(test['weather_class'].value_counts())

    # Save Weather Classwes
    ## Print Summary ##
    t_elapsed = datetime.datetime.now() - t_start

    # print(model_classify_sum.format(,
    #     t_elapsed,
    #     datetime.datetime.now()
    # ))

## Generator for Loading Each Part's Numpy Vectorized Padded Array ##
class partAllFramesDataGenerator(tf.keras.utils.Sequence):
    '''
    Generates data for Keras to load array from select numpy files and corresponding label stored in Pandas DataFrame
    '''

    def __init__(self, in_files, labels, batch_size=32, t_dim= 216, frame_dim=(32,96), n_classes=3, shuffle=True):
        '''
         part Data Generator Initialization
        '''

        # Generator Parameters #
        self.frame_dim = frame_dim
        self.t_dim = t_dim
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
        X = np.empty((self.batch_size,) + (self.t_dim,) + (1,) + self.frame_dim)

        # Generate Data #
        for i, file in enumerate(in_files):

            # Load All Frames from Data #
            frames = np.load(file)

            # Pad the Frame Array #
            add_length = self.t_dim - len(frames)
            pad_array = np.zeros( (add_length,) +  self.frame_dim )
            padded_frames = np.concatenate((pad_array, frames), axis=0)

            # Reshape each Frame for 1 Channel image for 2D Convultion #
            padded_frames = np.array([np.reshape(frame, (1,) + frame.shape) for frame in padded_frames])

            X[i,] = padded_frames

        return X

def int2weather_class(class_int):
    '''
    Convert the Class Integer back to Label
    '''
    if class_int == 0 :
        return 'Clear'
    if class_int == 1 :
        return 'Cloudy'
    if class_int == 2 :
        return 'Noisy'
    else :
        return 'Unknown'

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
    classify_pedestal_weather()
