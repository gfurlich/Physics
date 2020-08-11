#-------------------------------------------------------------------------------------#
#
#    File :                 fd_pedestal_weat_class_eval_model.py
#    Author :               Greg Furlich
#    Date Created :         2019-04-08
#
#    Purpose: To Evaluate the NN Model in classiying parts
#
#    Execution :   python fd_pedestal_weat_class_dnn.py
#
#---# Start of Script #---#

## Import Python Libraries ##

import numpy as np
import itertools
import os
import math

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Pandas
import pandas as pd

# Import Keras
# from keras.models import Sequential
from keras.utils import np_utils
# from keras.wrappers.scikit_learn import KerasClassifier
# from keras.utils.vis_utils import plot_model
from keras.models import model_from_json

# Import Sklearn
# from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import train_test_split

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
scratch_out = scratch+'out/'

# File Paths of Data for Generators #
all_files = scratch + 'Vectorized_Data/{0}_ped_fluct_vectorized.npy'
train_labels_hdf_file = scratch + 'DF/master_br_fd_ped_db_by_part_classified.csv'

# Infiles of Pandas DataFrames in local scratch space #
# training_classified_file = scratch+'DF/master_br_fd_ped_db_by_part_classified.csv'
master_fd_ped_db = scratch+'DF/master_fd_ped_db_by_part.h5'

# Out Models #
ml_weights_in = scratch_out + 'dnn_parameters.h5'
ml_model_in = scratch_out + 'dnn_model.json'

# Data Variables #
nrows, ncols = 2 * 16, 6 * 16

class_names = ['clear','cloudy','noisy']
n_classes = 3

## Evaluate Model ##
def eval_model():

    ## Load Training Data Labels and Make Pie Chart ##
    br_train_df = pd.read_csv(train_labels_hdf_file)
    s = br_train_df['part_weather_status'].value_counts()
    plot = s.plot.pie(y=s.index, labels=['Clear','Cloudy','Noisy','Missing'], colors=['royalblue','firebrick','seagreen','orange'], autopct='%1.0f%%', pctdistances=.75, labeldistance=1.1)
    plt.title('BR Training Data Set Weather Classification')
    plt.axis('equal')
    plt.savefig(scratch_out + 'BR_Training_Data_Classes.png')


    ## Load All Training Data DF ##
    # store_master_fd_ped_db = pd.HDFStore(master_fd_ped_db)
    # master_br_fd_ped_db_by_part = store_master_fd_ped_db.get('master_br_fd_ped_db_by_part')
    # # master_lr_fd_ped_db_by_part = store_master_fd_ped_db.get('master_lr_fd_ped_db_by_part')
    # store_master_fd_ped_db.close()
    #
    # br_nights = master_br_fd_ped_db_by_part['run_night']
    # n_all_parts = len(br_nights)

    ## All Br Data Generator ##

    ## Load Model to Evaluate ##

    ## Predict All BR Data ##

    ## Plot Predicted Data Pie Chart ##

    ## Plot Cumulated Dark Time With Model Classification of Clear ##

## Generator for Loading Each Part's Numpy Vectorized Padded Array ##
class partFlattenFramesDataGenerator(keras.utils.Sequence):
    '''
    Generates data for Keras to load array from select numpy files and corresponding label stored in Pandas DataFrame
    '''

    def __init__(self, in_files, labels, batch_size=32, flat_dim=(32*96), n_classes=3, shuffle=True):
        '''
         part Data Generator Initialization
        '''

        # Generator Parameters #
        self.flat_dim = frame_dim
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

            # Flatten frame from 2D to 2D #
            frame = np.array(frames[-1])
            frame = frame.flatten()

            X[i,] = frame

        return X


if __name__ == '__main__':
    eval_model()
