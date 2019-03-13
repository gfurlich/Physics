#-------------------------------------------------------------------------------------#
#
#    File :                 fd_pedestal_rnn_vectorization_v_chpc.py
#    Author :               Greg Furlich
#    Date Created :         2018-11-27
#
#    Purpose: To load in all fd pedestal preprocessed data into numpy array and pad Frames so they all have the same length
#
#    Execution :   python fd_pedestal_rnn_vectorization_v_chpc.py
#
#---# Start of Script #---#


## Import Python Libraries #
from __future__ import print_function
import os, math, sys, datetime, gc

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.colors as colors
import psutil

## Import Training Data File Information ##

# Parent Directories #
# parent = '/uufs/chpc.utah.edu/common/home/u0949991/'
# ml_parent =  parent + 'weat_ml/'

# CHPC Scratch Space #
# scratch = '/scratch/kingspeak/serial/u0949991/'
# scratch_out = scratch+'out/'

# CHPC Local Scratch Space #
scratch = '/scratch/local/u0949991/'
scratch_out = scratch+'out/'

# Infiles of Pandas DataFrames in local scratch space #
training_classified_file = scratch+'DF/master_br_fd_ped_db_by_part_classified.csv'
master_fd_ped_db = scratch+'DF/master_fd_ped_db_by_part.h5'
pmt_pos_file = scratch+'DF/pmt_positions.h5'

# Infiles of Pandas DataFrames in local scratch space #
load_ped_data_df = scratch+'Data/fd_ped_h5/{0}_ped_fluct.h5'

# Outfiles for numpy arrays :
save_ped_data_npy = scratch+'Data/fd_ped_vect/{0}_ped_fluct_vectorized_padded.npy'
save_ped_data_npy_nonpadded = scratch+'Data/fd_ped_vect_nonpadded/{0}_ped_fluct_vectorized.npy'

# All Data Vectorized and Padded #
missing_data = scratch_out+'insufficent_data.txt'

# Data Variables #
nrows, ncols = 2 * 16, 6 * 16

# Found Iteratively :
max_frames = 216

## Accessing Data ##
# all_training_data is list of numpy arrays with depth t for each part with a array of nrows by ncols for each frame
# all_training_data[i] selects the ith np array from lists
# all_training_data[i][j] selects the jth frame from the ith np array in the list
# all_training_data[i][j][k] selects the kth row of the jth frame from the ith np array in the list
# all_training_data[i][j][k][l] selects the lth column of the kth row of the jth frame from the ith np array in the list

def main():

    print( 'Pandas:'+pd.__version__, 'Numpy:'+np.__version__,'Mlp:'+mpl.__version__)

    print('Loading FD Part Data Bases...')

    ## Load All Preproccessed Pedestal Data Nights DataFrame ##
    store_master_fd_ped_db = pd.HDFStore(master_fd_ped_db)
    master_br_fd_ped_db_by_part = store_master_fd_ped_db.get('master_br_fd_ped_db_by_part')
    # master_lr_fd_ped_db_by_part = store_master_fd_ped_db.get('master_lr_fd_ped_db_by_part')
    store_master_fd_ped_db.close()

    br_nights = master_br_fd_ped_db_by_part['run_night']
    n_all_parts = len(br_nights)
    # print(list(master_br_fd_ped_db_by_part), n_all_parts)

    ## Load PMT Position DATA ##
    load_pmt_positions_file = pd.HDFStore(pmt_pos_file)
    pmt_positions = load_pmt_positions_file.get('pmt_positions')
    load_pmt_positions_file .close()

    # Read in PMT Positions and Convert back to rows and cols :
    pmt_x, pmt_y = pmt_positions['pmt_x'].tolist(), pmt_positions['pmt_y'].tolist()
    global row, col
    row = np.asarray(pmt_y) / 1.5
    col = np.rint(np.asarray(pmt_x) / math.sqrt(3) - .1)    # some bug keeps flooring 50 to 51...
    row = row.astype(int)
    col = col.astype(int)

    ## Load All Frames of Every Part in Data Set ##
    print('Loading All FD Pedestal Data for Vectorization and Padding...')

    # all_data = []

    f = open(missing_data,'w+')

    # for i in range(1000):
    for i in range(n_all_parts):

        # Set Date and Part to Load :
        ymds = master_br_fd_ped_db_by_part['run_night'].iloc[i].strftime('%Y-%m-%d').split('-')
        ymds.append('0')# Add BR
        ymds = 'y{0}m{1}d{2}s{3}'.format(ymds[0], ymds[1], ymds[2], ymds[3])
        part = master_br_fd_ped_db_by_part['part'].iloc[i]

        # print('Loading and Padding FD Pedestal Data part: {0} {1} ({2}/{3})'.format(ymds, part, i, n_all_parts))
        if (i % 100) == 0:
            print('Loading and Padding FD Pedestal Data part: {0} {1} ({2}/{3})'.format(ymds, part, i, n_all_parts))
            print_process_mem()

        if (i % 1000) == 0:
            print_mem_info()

        X, part_status = _vectorize_pedestal_data_df(ymds, part)

        # Check for Part Status for missing data :
        if part_status is False:
            f.write('Insufficient Data from {0} {1}\n'.format(ymds, part))

        # Clear Memory #
        del ymds, part
        del X
        gc.collect()

    # print('')
    f.close()
    #
    # print('Saving All Vectorized and Padded FD Pedestal Data as Numpy Arrays to {0}'.format(weather_ml_data))
    # np.save(weather_ml_data, all_data)
    # print('File Saved...')
    # del all_data

def _vectorize_pedestal_data_df(ymds, part):
    '''
    Load the Pedestal Data from Pandas DataFrame from HDF5 File containing pedestal fluctation data by each part.
    '''

    # Load HDF5 File :
    store_df = pd.HDFStore(load_ped_data_df.format(ymds))

    # Load DataFrame from HDF5 file :
    ped_fluct_df = store_df.get('ped_fluct_df')
    frame_info_df = store_df.get('frame_info_df')
    part_frames = frame_info_df.index[frame_info_df['frame_part'] == part]
    nframes = len(part_frames)

    # Close HDF5 File :
    store_df.close()

    # Pre-allocate Loaded Data Array and Padding Array :
    ped_fluct_data = np.empty((nframes, nrows, ncols))
    add_length = max_frames - nframes
    padded_zero_array = np.zeros((add_length, nrows, ncols))

    for frame in range(nframes):

        # print part, ymds, frame, part_frames[frame], part_frames[-1]

        # Load Data, Format, and Normalize
        X = np.empty((nrows,ncols))
        ped_fluct = ped_fluct_df.iloc[part_frames[frame]].values  # Convert to numpy array

        # Absolute Value :
        ped_fluct = np.fabs(ped_fluct)

        # Reorder to match row and column index to create Frame Array :
        for i in range(len(ped_fluct)):

            # X[row[i], col[i]] = ped_fluct[i]

            # Add offset for asthetics :
            if ped_fluct[i] != 0 :
                X[row[i], col[i]] = ped_fluct[i]
            else :
                X[row[i], col[i]] = 1

        # Normalize Pedestal Data :
        norm = colors.LogNorm(vmax = X.max(), vmin = .1)
        # norm = colors.Normalize(vmax = X.max(), vmin = X.min())
        X = norm(X)

        # Flip array so Bottom Left of Image is bottom left of array
        X = np.flip(X, 0)

        # Appended New Vectorized Frame
        ped_fluct_data[frame] = X

        # Rough Plot of Frame for checking :
        # plt.imshow(X, cmap='inferno', vmax=X.max(), vmin=.1) # Log Norm
        # plt.imshow(X, cmap='inferno', vmax=X.max(), vmin=X.min())
        # plt.show()

        # Clean Up :
        del X

    # Pre-allocate Padded Array and Concatenate to Fill :
    # Pad Pedestal Data with Zero Arrays in Front :
    padded_ped_fluct_data = np.empty((max_frames, nrows, ncols))
    ped_fluct_data = np.asarray(ped_fluct_data)
    padded_ped_fluct_data = np.concatenate((padded_zero_array, ped_fluct_data), axis=0)

    # Save Padded and Nonpadded pedestal Data as Numpy Arrays #
    print('Saving Vectorized and Padded FD Pedestal Data as Numpy Arrays to {0}...'.format(save_ped_data_npy.format(ymds,part)))
    np.save(save_ped_data_npy_nonpadded.format(ymds,part), ped_fluct_data)
    np.save(save_ped_data_npy.format(ymds,part), padded_ped_fluct_data)

    # Check Last Frame if zero for insufficient Pedestal Data to determine part Status #
    # If empty :
    if (np.array_equal(padded_ped_fluct_data[-1], np.zeros((nrows,ncols))) == True ):
        # print(nframes, ped_fluct_data.shape, ymds, part)
        part_status = False
    # If non-empty :
    else :
        part_status = True

    # Clean up :
    del ped_fluct_data, padded_zero_array, add_length
    # del padded_ped_fluct_data
    del ped_fluct_df, frame_info_df, part_frames, nframes
    # print_process_mem()

    return padded_ped_fluct_data, part_status

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
    main()
