#-------------------------------------------------------------------------------------#
#
#    File :                 weather_class_train_model_v_chpc_cnn.py
#    Author :               Greg Furlich
#    Date Created :         2019-01-14
#
#    Purpose: To Classifiy Weather of parts into clear, cloudy, and bad given the first snapshot of each part using a CNN on the U of U CHPC
#
#    Execution :   python weather_class_train_model_v_chpc_cnn.py
#
#
#---# Start of Script #---#

## Import Python Libraries ##

import numpy as np
import itertools
import os
import math
import psutil
# import sys
# from scipy import stats
# import datetime
# from ROOT import TCanvas, TGraph, TH1F, TF1, TH1D

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

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
scratch_out = scratch+'out/'
scratch_vect_data = scratch+'Vectorized_Data/'

# Infiles of Pandas DataFrames in local scratch space #
training_labels = scratch_vect_data + 'weather_training_labels_v_rnn.npy'
training_vect_data = scratch_vect_data + 'weather_training_data_v_rnn.npy'

# Outfiles for Saving ML Model :
# cnn_model = scratch_out + 'fd_ped_cnn_model.json'
ml_weights_out = scratch_out + 'weather_ml_weights_v_cnn.h5'
ml_model_out = scratch_out + 'weather_ml_model_v_cnn.json'

# Data Variables #
nrows, ncols = 2 * 16, 6 * 16

# Found Iteratively :
max_frames = 216

class_names = ['clear','cloudy','noisy']
n_classes = 3

def weather_class():

    # print_process_mem()
    # print_mem_info()

    print 'Loading Vectorizated Training Data...'

    # Print Tensor Flow Version :
    #print(tf.__version__, pd.__version__)

    ## Load Vectorized Data ##
    training_data = np.load(training_vect_data)
    n_train_parts = len(training_data)

    ## Load Label Data ##
    onehot_y = np.load(training_labels)
    y = np.argmax(onehot_y, axis=-1)
    y = y.astype('int8')

    # print_process_mem()
    # print_mem_info()

    ## Prepare Input into Model ##

    print training_data.shape, training_data.shape[0]

    # Get Last Frame from Padded Vectorized Frame Data #
    X = [ training_data[i][-1] for i in range(n_train_parts)]
    X = np.asarray(X)

    print X.shape

    X_len = len(X)
    X = X.reshape(X_len, nrows, ncols, 1)

    print len(X), len(y), len(onehot_y)
    print X[0].shape, y[0].shape, onehot_y[0].shape
    print X.shape, y.shape, onehot_y.shape

    ## Encode classifiers to one-hot ##
    # encoder = LabelEncoder()
    # encoder.fit(y)
    # encoded_y = encoder.transform(y)
    # onehot_y2 = np_utils.to_categorical(encoded_y)
    #
    # print onehot_y, onehot_y2

    print_process_mem()
    print_mem_info()

    ## Split the Traing Set into  Training Set and Validation Set #

    # fix random seed for reproducibility :
    # seed = 20181101
    # np.random.seed(seed)
    #
    # # split into 67% for train and 33% for test :
    # # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=seed)
    #
    #
    # ## Construct Model ##
    # print 'Constructing CNN Model...'
    #
    # ## Cross Validation ##
    # nfolds = 3 # 1/3 used for validation
    # nrepeats = 1
    # nepochs = 50
    #
    # kfold = RepeatedStratifiedKFold(n_splits=nfolds, n_repeats=nrepeats, random_state=seed)
    # model_scores = []
    # fold = 1
    # best_score = 0
    #
    # ## Train Model ##
    # print 'Training Model with {0}-fold split over {1} repeats ({2} iterations)...'.format(nfolds, nrepeats, nrepeats * nfolds)
    #
    # for train, val in kfold.split(X, y):
    #
    #     # print train, val
    #     # print len(train), len(val)
    #     # print len(X[train]), len(onehot_y[train]), X[train[0]], onehot_y[train[0]]
    #
    #     # Construct model :
    #     model = keras.Sequential([
    #
    #         # Input Layer and Convolution Layer :
    #         keras.layers.Conv2D(32, (4,4), input_shape=( nrows, ncols, 1), activation='relu'),
    #         keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #         keras.layers.Conv2D(32, (4,4), activation='relu'),
    #         keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #
    #         # Flatten Layer :
    #         keras.layers.Flatten(),
    #
    #         # Dense Layers :
    #         keras.layers.Dense(48, activation='relu'),
    #         keras.layers.Dense(12, activation='relu'),
    #         keras.layers.Dense(6, activation='relu'),
    #
    #         # Output Layer :
    #         keras.layers.Dense(n_classes, activation='softmax')
    #
    #     ])
    #
    #     # print ( float(len(val)) / float(len(train)+len(val))) * 100
    #
    #     # Compile model :
    #     model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    #
    #     # Fit the model :
    #     history = model.fit(X[train], onehot_y[train], validation_data=(X[val], onehot_y[val]), epochs=nepochs, verbose=0)
    #
    # 	# Evaluate the model :
    #     scores = model.evaluate(X[val], onehot_y[val], verbose=0)
    #
    #     # print train[0:10], val[0:10]
    #     print 'Validation Accurracy of Model {0}/{1} (train size={2}, test size={3}): {4:.2f}% Loss:  {5:.2f}'.format( fold, nrepeats * nfolds, len(train), len(val), scores[1]*100,  scores[0])
    #     model_scores.append(scores[1] * 100)
    #
    #     # Save 'best' model :
    #     if scores[1] > best_score :
    #         best_score = scores[1]
    #         best_history = history
    #
    #         # print 'Saving Trained Model to {0}'.format(ml_model_out)
    #         # print 'Saving Trained Weights to {0}'.format(ml_weights_out)
    #
    #         # serialize model to JSON
    #         model_json = model.to_json()
    #         with open(ml_model_out, 'w') as json_file:
    #             json_file.write(model_json)
    #
    #         # serialize weights to HDF5
    #         model.save_weights(ml_weights_out)
    #
    #         ## Make confusion matrix ##
    #         y_pred = model.predict(X)
    #         y_classes = np.argmax(y_pred, axis=-1)
    #         cnf_matrix = confusion_matrix(np.array(y), np.array(y_classes))
    #
    #         y_val_pred = model.predict(X[val])
    #         y_val_classes = np.argmax(y_val_pred, axis=-1)
    #         cnf_val_matrix = confusion_matrix(np.array(y[val]), np.array(y_val_classes))
    #
    #         ## Plot and Save confusion matrix ##
    #         mpl.style.use('classic')
    #         plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='CNN Training Data Normalized Confusion Matrix')
    #         plt.savefig(scratch_out+'weather_class_cnn_cfm_norm.png', bbox_inches='tight')
    #         plt.clf()
    #
    #         plot_confusion_matrix(cnf_val_matrix, classes=class_names, normalize=True, title='CNN Validation Set Normalized Confusion Matrix')
    #         plt.savefig(scratch_out+'weather_class_cnn_cfm_val_norm.png', bbox_inches='tight')
    #         plt.clf()
    #
    #         plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False, title='CNN Training Data Confusion Matrix')
    #         plt.savefig(scratch_out+'weather_class_cnn_cfm.png', bbox_inches='tight')
    #         plt.clf()
    #
    #         plot_confusion_matrix(cnf_val_matrix, classes=class_names, normalize=False, title='CNN Validation Set Confusion Matrix')
    #         plt.savefig(scratch_out+'weather_class_cnn_cfm_val.png', bbox_inches='tight')
    #         plt.clf()
    #
    #     fold+=1
    #
    # ## Evaluate Model ##
    # print 'Evaluating all Models...'.format(nrepeats, nfolds)
    #
    # model_scores = np.asarray(model_scores)
    #
    # const, mu, sigma, econst, emu, esigma, chi2, ndf, prob = _eval_folds(model_scores, nfolds, nrepeats)
    # # plt.savefig(ml_parent+'kfold_model_accuracy_dist.png', bbox_inches='tight')
    # # plt.clf()
    #
    # # print'Mean Validation acc: {0:.2f}% +/- {1:.2f} %'.format(np.mean(model_scores), np.std(model_scores))
    # print'Model Validation Accuracy Distribution: mean {0:.2f}, sigma {1:.2f}, chi2/ndf {2:.2f}/{3}, p_value {4:.2f} '.format(mu, sigma, chi2, ndf, prob)
    #
    # # history = model.fit(X_train, onehot_y_train, validation_data=(X_val,onehot_y_val), epochs=50)
    #
    # ## Plotting the 'best' model accuracy progression ##
    #
    # acc = best_history.history['acc']
    # val_acc = best_history.history['val_acc']
    # loss = best_history.history['loss']
    # val_loss = best_history.history['val_loss']
    #
    # epochs = range(1, len(acc) + 1)
    #
    # ## Loss Plot ##
    # mpl.style.use('seaborn')
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.plot(epochs, loss, 'bo', label='Training Loss')
    # plt.plot(epochs, val_loss, 'g', label='Validation Loss')
    # plt.title('CNN Training and Validation Loss')
    # ax.set_ylim(bottom=0)
    # ax.set_xlim(0,nepochs)
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # # plt.text(0.45, 0.5, 'Preliminary', horizontalalignment='center', verticalalignment='center', color='gray', fontsize=28, fontweight='bold', rotation=30, transform=ax.transAxes)
    # plt.savefig(scratch_out+'weather_class_train_cnn_accur.png', bbox_inches='tight')
    # plt.clf()
    #
    # ## Plot Training Accuracy Plot ##
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.plot(epochs, acc, 'bo', label='Training Accuracy')
    # plt.plot(epochs, val_acc, 'g', label='Validation Accuracy')
    # plt.title('CNN Training and Validation Accuracy')
    # ax.set_ylim(0, 1)
    # ax.set_xlim(0,nepochs)
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # # plt.text(0.45, 0.5, 'Preliminary', horizontalalignment='center', verticalalignment='center', color='gray', fontsize=28, fontweight='bold', rotation=30, transform=ax.transAxes)
    # plt.savefig(scratch_out+'weather_class_train_cnn_loss.png', bbox_inches='tight')
    # plt.clf()

# Gaussian Fit Function :
def gaus(x, const, mu, sigma):
    return const* np.exp(-0.5*((x - mu)/sigma)**2)

def _eval_folds(model_scores, nfolds, nrepeats):

    # Define Bin Size #
    xmin = np.floor(model_scores.min())
    xmax = np.ceil(model_scores.max())
    nbins = int((xmax-xmin)*10)
    print xmin, xmax

    # Create Python Histogram
    hist, bin_edges, patches = plt.hist(model_scores,nbins,(xmin,xmax),color='g',alpha=0.6)
    bin_centers = (bin_edges[1:]+bin_edges[:-1])/2.

    # Find Non-zero bins in Histogram
    nz = hist>0

    # Plot the Model Distribution and Fit
    root_hist = np.zeros(nbins+2,dtype=float)
    root_hist[1:-1] = hist
    h = TH1D('h','hist',nbins,bin_edges)
    h.SetContent(root_hist)

    # Fit histogram with root
    h.Fit('gaus','','',xmin,xmax)

    # Get Root Fit and Goodness of Fit Parameters #
    f = h.GetFunction('gaus')
    const,mu,sigma = f.GetParameter(0), f.GetParameter(1), f.GetParameter(2)
    econst,emu,esigma = f.GetParError(0), f.GetParError(1), f.GetParError(2)
    ndf,chi2,prob = f.GetNDF(),f.GetChisquare(),f.GetProb()

    # Define Fit Curve #
    x = bin_centers
    root_gaus = (const,mu,sigma)
    f = gaus(x,*root_gaus)

    # Draw Histogram and Fit with Python Mathplotlib :
    fig, ax = plt.subplots()
    plt.hist(model_scores,nbins,(xmin,xmax),color='g',alpha=0.6)
    plt.plot(bin_centers, f, 'k--', linewidth=2)
    plt.xlim(xmin, xmax)

    # Plot Text Boxes #
    root_txt = '\n'.join((
        r'PyROOT Fit:',
        r'$n = {0}$'.format(len(model_scores)),
        # r'$height={0:.4f} \pm {1:.4f}$'.format(const, econst),
        r'$\mu = {0:.4f} \pm {1:.4f}$'.format(mu, emu),
        r'$\sigma = {0:.4f} \pm {1:.4f}$'.format(sigma, esigma),
        r'$\chi^2 / ndf = {0:.4f} / {1}$'.format(chi2, ndf),
        r'$prob = {0:.4f}$'.format(prob)))

    ax.text(0.05, 0.95, root_txt, transform=ax.transAxes, fontsize=12, verticalalignment='top')
    plt.title('CNN {0}-Fold Validation Accuracy'.format(nfolds, nrepeats))
    plt.xlabel('Validation Accuracy')
    plt.ylabel('Counts')
    plt.grid()
    plt.savefig(ml_parent+'CNN_kfold_distribution_validation_Acc_fit.png')

    return const, mu, sigma, econst, emu, esigma, chi2, ndf, prob

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
            plt.text(j, i, '{0:.0f}%'.format(cm[i, j] * 100), horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')
        else :
            plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')

    # plt.text(0.33, 0.5, 'Preliminary', horizontalalignment='center', verticalalignment='center', color='gray', fontsize=28, fontweight='bold', rotation=45, transform=ax.transAxes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

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
