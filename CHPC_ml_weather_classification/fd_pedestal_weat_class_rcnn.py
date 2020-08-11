#-------------------------------------------------------------------------------------#
#
#    File :                 fd_pedestal_weat_class_rcnn.py
#    Author :               Greg Furlich
#    Date Created :         2019-03-22
#
#    Purpose: To Classifiy Weather of parts into clear, cloudy, and bad given the padded snapshots of each part
#
#    Execution :   python fd_pedestal_weat_class_rcnn.py
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
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense
from keras.layers import InputLayer
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils.vis_utils import plot_model
from keras.models import model_from_json

# Import Sklearn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold

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
scratch_out = scratch+'RCNN_out/'

# File Paths of Data for Generators #
file = scratch + 'RCNN_Vectorized_Data/{0}_ped_fluct_vectorized_padded.npy'
labels_hdf_file = scratch + 'RCNN_DF/master_br_fd_ped_db_by_part_classified.csv'

# Out Models #
ml_weights_out = scratch_out + 'rcnn_parameters.h5'
ml_model_out = scratch_out + 'rcnn_model.json'

# Data Variables #
nrows, ncols = 2 * 16, 6 * 16
t_dim = 216

class_names = ['clear','cloudy','noisy']
n_classes = 3

# RCNN Model Options #
nepochs = 50
model_loss = keras.losses.categorical_crossentropy
model_metric = keras.metrics.categorical_accuracy
# model_optimizer = keras.optimizers.SGD()
# model_optimizer = keras.optimizers.RMSprop()
# model_optimizer = keras.optimizers.Adagrad()
# model_optimizer = keras.optimizers.Adadelta()
# model_optimizer = keras.optimizers.Adam()
# model_optimizer = keras.optimizers.Adamax()
# model_optimizer = keras.optimizers.Nadam()

# Adagrad seems to be the most stable of Optimizers #
model_optimizer = keras.optimizers.Adagrad(lr=0.0085, epsilon=None, decay=0)

## Print Summary ##
model_sum = '''
== RCNN Optimizer INFO ==========================

Training Epochs =\t{0}

Training Loss :\t{1}
Training Optimizer :\t{2}
Training Metrics :\t{3}

Validation Accuracy =\t{4}
Validation Cross Entropy =\t{5}

RCNN Model Process Time :\t{6}
RCNN Model Process Date :\t{7}

== RCNN Optimizer INFO ==========================
'''

def weather_class():

    # Start Timer #
    t_start = datetime.datetime.now()

    # Check Memory #
    print_process_mem()
    print_mem_info()

    # Print GPU Info #
    print(device_lib.list_local_devices())

    # Print Tensor Flow Version :
    #print(tf.__version__, pd.__version__)

    ## List Training Data Files and Load Labels DataFrame #

    # Load BR Weather info CSV to Pandas DataFrame
    df = pd.read_csv(labels_hdf_file)

    # Create Index data and Remove Missing Data #
    site = '0'
    ymdsp = df[df['part_weather_status'] != 'Missing'].apply(lambda row : 'y{0}m{1}d{2}s{3}p{4}'.format(row['run_night'].split('-')[0], row['run_night'].split('-')[1], row['run_night'].split('-')[2], site, row['part']), axis=1)
    print('Found {0} Non Missing Parts out of {1} Parts Made {2} Indexes'.format(len(df[df['part_weather_status'] != 'Missing']), len(df), len(ymdsp)))


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
    train_gen = partAllFramesDataGenerator(in_files = train_files, labels = train_labels, batch_size=16)
    val_gen = partAllFramesDataGenerator(in_files = val_files, labels = val_labels, batch_size = 16)

    ## Training Data ##
    val_X, val_y = partAllFramesDataGenerator(in_files = val_files, labels = val_labels, batch_size = len(val_files), shuffle=False).__getitem__(0)

    print_process_mem()
    print_mem_info()

    ## RCNN Model ##

    # Validation Set Shape #
    # X_shape, y_shape = train_gen.__getitem__(0)
    # X_shape, y_shape = X_shape.shape, y_shape.shape
    # print 'Input Train {0} and Label {1} shapes.format(X_shape, y_shape)

    # Construct model #
    print('Constructing RCNN Model...')

    model = keras.Sequential([

        # First Convolution Layer with pooling and batch normalization #
        keras.layers.TimeDistributed(keras.layers.Conv2D(8, (4,4), activation='relu', data_format='channels_first'), input_shape=(t_dim, 1, nrows, ncols)),
        # keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_first')),
        # keras.layers.TimeDistributed(keras.layers.BatchNormalization()),
        keras.layers.TimeDistributed(keras.layers.Dropout(0.4)),

        # Second Convolution Layer with pooling and batch normalization #
        keras.layers.TimeDistributed(keras.layers.Conv2D(8, activation='relu', kernel_size=(4, 4), data_format='channels_first')),
        # keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_first')),
        # keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.4),

        # Flatten Frame Spatial Dimensions to dense
        keras.layers.TimeDistributed(keras.layers.Flatten()),

        # First LSTM Layer #
        keras.layers.CuDNNLSTM(48, return_sequences=True),
        keras.layers.Dropout(0.4),

        # Second LSTM Layer #
        # keras.layers.CuDNNLSTM(12, return_sequences=True),
        # keras.layers.Dropout(0.3),

        # Third LSTM Layer #
        keras.layers.CuDNNLSTM(6),
        keras.layers.Dropout(0.4),

        # Categorize Data #
        keras.layers.Dense(n_classes, activation='softmax')

        ])

    # Compile Model #
    model.compile(loss=model_loss , optimizer=model_optimizer, metrics=[model_metric])

    # RCNN Optimizer Tests ##

    # Print Model Summary #
    print(model.summary())

    print_process_mem()
    print_mem_info()

    # Fit Model with Test Generator #
    print('Training RCNN Model...')
    history = model.fit_generator(generator=train_gen, validation_data=val_gen, epochs=nepochs, verbose=0)

	# Evaluate Model with validation Generator #
    print('Evaluating RCNN Model...')
    scores = model.evaluate_generator(val_gen, verbose=0)

    # print(scores)
    print('Validation Accuracy of RCNN Model : {0:.2f}% Cross Entropy:  {1:.2f}'.format( scores[1]*100,  scores[0]))

    ## Plotting model performance ##
    print('Plotting RCNN Model Performance...')

    # print(history.history.keys())

    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    ## Training Loss Plot ##
    mpl.style.use('ggplot')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(epochs, loss, 'bo', label='Training Set')
    plt.plot(epochs, val_loss, 'g', label='Validation Set')
    plt.title('RCNN Cross Entropy')
    ax.set_ylim(bottom=0)
    ax.set_xlim(0,nepochs)
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy')
    plt.legend()
    # plt.text(0.45, 0.5, 'Preliminary', horizontalalignment='center', verticalalignment='center', color='gray', fontsize=28, fontweight='bold', rotation=30, transform=ax.transAxes)
    plt.savefig(scratch_out+'rcnn_crossentropy.png', bbox_inches='tight')
    plt.clf()

    ## Training Accuracy Plot ##
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(epochs, acc, 'bo', label='Training Set')
    plt.plot(epochs, val_acc, 'g', label='Validation Set')
    plt.title('RCNN Accuracy')
    ax.set_ylim(0, 1)
    ax.set_xlim(0,nepochs)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    # plt.text(0.45, 0.5, 'Preliminary', horizontalalignment='center', verticalalignment='center', color='gray', fontsize=28, fontweight='bold', rotation=30, transform=ax.transAxes)
    plt.savefig(scratch_out+'rcnn_accuracy.png', bbox_inches='tight')
    plt.clf()

    ## Validation Confusion Matrix ##
    val_y_pred = model.predict(val_X)
    val_y_pred_classes = np.argmax(val_y_pred, axis=-1)
    val_y_true_classes = np.argmax(val_y, axis=-1)
    print(val_y_true_classes.shape, val_y_pred_classes.shape)
    cnf_matrix = confusion_matrix(np.array(val_y_true_classes), np.array(val_y_pred_classes))

    ## Plot and Save confusion matrix ##
    mpl.style.use('classic')
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='RCNN Normalized Confusion Matrix')
    plt.savefig(scratch_out+'rcnn_val_cfm_norm.png', bbox_inches='tight')
    plt.clf()

    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False, title='RCNN Confusion Matrix')
    plt.savefig(scratch_out+'rcnn_val_cfm.png', bbox_inches='tight')
    plt.clf()

    ## Save CNN Model ##
    print('Saving Trained RCNN Model to {0}'.format(ml_model_out))
    print('Saving Trained RCNN Weights to {0}'.format(ml_weights_out))

    # serialize model to JSON
    model_json = model.to_json()
    with open(ml_model_out, 'w') as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(ml_weights_out)

    ## Print Summary ##
    t_elapsed = datetime.datetime.now() - t_start

    # '''
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
    #
    # == RCNN Optimizer INFO ==========================
    # '''

    print(model_sum.format(
        nepochs,
        str(model_loss),
        str(model_optimizer),
        str(model_metric),
        scores[1]*100,
        scores[0],
        t_elapsed,
        datetime.datetime.now()
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

    if normalize:
        cax = plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
        plt.title(title)
        cbar = plt.colorbar(cax, ticks = [0,.25,.5,.75,1])
        tick_marks = np.arange(len(classes))
        cbar.ax.set_yticklabels(['0','.25','.5','.75','1'])
        # cbar.ax.set_yticklabels(['0%','25%','50%','75%','100%'])
        # cbar.set_label('Label Accuracy Percent', rotation=270)

    else:
        im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        cbar = plt.colorbar(im)
        tick_marks = np.arange(len(classes))

    # plt.xticks(tick_marks, classes, rotation=45)
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.tick_params(axis='both', which='both')

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
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.tight_layout()

## Generator for Loading Each Part's Numpy Vectorized Padded Array ##
class partAllFramesDataGenerator(keras.utils.Sequence):
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

            # Reshape each Frame for 1 Channel image for 2D Convultion #
            frames = np.array([np.reshape(frame, (1,) + frame.shape) for frame in frames])

            X[i,] = frames

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
