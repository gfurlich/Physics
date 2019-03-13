#!/usr/bin/python
#-------------------------------------------------------------------------------------#
#
#    File :                 fd_pedestal_animation.py
#    Author :               Greg Furlich
#    Date Created :         2018-07-12
#
#    Purpose: To animate the pedestal values of BR and LR to see the night sky video. Based on fd_pedestal_animation.py but optimized for hex pixel and capture pedestal flucuations by taking differenceof pedestal values between subsequent minutes
#
#
#    Execution :   python fd_pedestal_animation.py -n <YYYYMCDD> -s <site>
#    Example :     python fd_pedestal_animation.py -n 20170225 -s 0
#
#    Sources : see pedstar.C for Tom Stroman's scripts for converting BR and LR data into animations
#
#---# Start of Script #---#

## Importing Python Modules ##
import argparse
import re
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.patches import RegularPolygon
import errno

## Defining Global Variables of Date and Pedestal Data ##

YMD = re.compile('(\d{4})(\d{2})(\d{2})')

FD_PED_DATA = {
    #0: '/ssd32g/rtdata/calibration/fdped/black-rock/y{0}m{1}d{2}.ped.dst.gz',
    #1: '/ssd32g/rtdata/calibration/fdped/long-ridge/y{0}m{1}d{2}.ped.dst.gz',
    0: '/home/gfurlich/TAResearch/FD_Ped_Weather/fdped/black-rock/y{0}m{1}d{2}.ped.dst.gz',
    1: '/home/gfurlich/TAResearch/FD_Ped_Weather/fdped/black-rock/y{0}m{1}d{2}.ped.dst.gz',
}


OUT_DIR = {
    'frame': '/home/gfurlich/TAResearch/FD_Ped_Weather/Frame/{0}_dh/',
    'gif': '/home/gfurlich/TAResearch/FD_Ped_Weather/GIF/',
}

OUT_DATA = {
    'fluct': '/home/gfurlich/TAResearch/FD_Ped_Weather/Data/{0}_ped_fluct.csv',                # Save Pedestal Fluctuation
    'fluct_norm': '/home/gfurlich/TAResearch/FD_Ped_Weather/Data/{0}_ped_fluct_norm.csv',      # Save Normalized Pedestal fluctuations
    'fluct_info': '/home/gfurlich/TAResearch/FD_Ped_Weather/Data/{0}_ped_fluct_info.csv',      # Save Pedestal Frame Part and Minute
}

OUT_FIG = {
    'frame': 'frame_p{:02d}_min{:02d}.jpg',
    'gif': '{0}_dh.gif',
}

# Hexagonal Pixel Radius:
# Hexagonal Pixel radius to each vertex' WRONG'. Used to calculate position of centers
r = 1
r_true = .75 * r    # scale r so that edges of hexagons don't overlap

# Difference in Hexagon's Centers :
dx = math.sqrt(3) * r    # Difference of x between row centers
dy = 1.5 * r        # Difference of y between column centers

# BR and LR columns and rows of pmts
nrows, ncols = 32, 96

## Initialize Frame Variables ##
night_sky = plt.figure()
#ax = night_sky.gca()
#frame_text = ax.text(2*r,2*r,'',color='white', fontsize=8)

# Axis Options :
#ax.set_aspect('equal')
#ax.set_xticks([])
#ax.set_yticks([])
#ax.set_axis_bgcolor('black')

# Frame Limits :
#plt.xlim([-r, ncols*dx])        # X Range
#plt.ylim([-r, (nrows)*dy])        # Y Range

cmap = cm.inferno

def animate_fd_pedestal(night, site):

    # Data Selection :
    y, m, d = YMD.findall(str(night))[0]
    global ymds
    ymds = 'y{0}m{1}d{2}s{3}'.format(y, m, d, site)

    # Temp dump File for I/O
    temp_dump_file = 'tempdump.txt'

    # Dump Pedestal DST to temp ASCII File
    _dump_pedestal_dst(night, site,temp_dump_file)

    # Import Pedestal data :
    fd_ped_data = _import_ped_txt(temp_dump_file)

    # Process data :
    all_ped_fluct_data, ped_fluct_norm, fluct_part, fluct_minute, nframes = _preprocess_fd_pedestal_data(fd_ped_data)

    # Create animation :
    _create_pedestal_animation(all_ped_fluct_data, ped_fluct_norm, fluct_part, fluct_minute, nframes)

def _create_pedestal_animation(all_ped_fluct_data, ped_fluct_norm, fluct_part, fluct_minute, nframes):

    print 'Generating {0} Frames'.format(nframes)
    ## Define Global Variables for Frame Creation ##

    # Load PMT Position :
    global g_pmt_x, g_pmt_y
    pmt_x, pmt_y = _load_pmt_positions()
    g_pmt_x, g_pmt_y = pmt_x.tolist(), pmt_y.tolist()

    global g_all_ped_fluct_data, g_ped_fluct_norm, g_fluct_part, g_fluct_minute
    g_all_ped_fluct_data, g_ped_fluct_norm, g_fluct_part, g_fluct_minute = all_ped_fluct_data, ped_fluct_norm, fluct_part, fluct_minute

    ## Create Frames ##
    '''
    ## Test for one frame ##
    frame = 0

    __create_frame_diff_hex_pixel(frame)
    '''

    # Select Part :
    for frame in range(0,nframes):

        #print 'Rendering Pedestal Sky View for frame: {:02d} / {:02d}\r'.format(int(frame), int(nframes))

        # Render Frame with desired Pixel Type :
        __create_frame_diff_hex_pixel(frame)

    ## Create GIF of Night From Frames using ImageMagick ##

    out_gif = OUT_DIR['gif'] + OUT_FIG['gif'].format(ymds)
    out_dir = OUT_DIR['frame'].format(ymds)

    print 'Rendering Pedestal Fluctuation GIF : '+out_gif

    # Use ImageMagick and System commands:
    #os.system('convert -delay 5 -loop 0 -layers Optimize'+out_dir+'*.jpg '+out_gif)
    os.system('convert '+out_dir+'*.jpg '+out_gif)

def __create_frame_diff_hex_pixel(frame):
    '''
    Create animation frame using custom python hexagonal pixel color map and subracting the difference between each minute.
    '''

    ax = night_sky.gca()
    #frame_text = ax.text(2*r,2*r,'',color='white', fontsize=8)

    # Axis Options :
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_bgcolor('black')

    # Frame Limits :
    plt.xlim([-r, ncols*dx])        # X Range
    plt.ylim([-r, (nrows)*dy])        # Y Range

    # Get Minute and Part information :
    minute, part = g_fluct_minute[frame], g_fluct_part[frame]

    '''
    # Plot Each Hexagon :
    for xx, yy, c in zip(pmt_x, pmt_y, ped_fluct_norm[frame]):
        #print xx, yy, c
        ax.add_artist(RegularPolygon(xy=(xx, yy),
            numVertices=6,
            radius=r_true,
            color = c ) )
    '''

    # Hexagon Scatter Plot:
    # ax.scatter(pmt_x, pmt_y, s=(r * 5.2)**2, c = ped_fluct_norm[frame], marker = 'h', edgecolor='none')
    ax.scatter(g_pmt_x, g_pmt_y, s=(r * 5.2)**2, c = cmap(g_ped_fluct_norm[frame]), marker = 'h', edgecolor='none')

    # Add text information to plot :
    #plt.text(2*r,2*r,r'part: {:02d} $\Delta$ mins: {:02d} - {:02d}'.format(int(part),int(minute),int(minute+1)),color='white', fontsize=8)
    plt.text(2*r,2*r,r'part: {:02d}'.format(int(part)),color='white', fontsize=14)

    # Save Figure Name :
    out_fig = OUT_DIR['frame'].format(ymds) + OUT_FIG['frame'].format(int(part),int(minute))

    # Save Plot :
    #night_sky.savefig(out_fig, dpi = 200, bbox_inches='tight', facecolor = 'black', pad_inches=-.01)
    night_sky.savefig(out_fig, dpi = 40, bbox_inches='tight', facecolor = 'black', pad_inches=-.01)

    night_sky.clf()

def _preprocess_fd_pedestal_data(fd_ped_data):

    print 'Pre-Processing Pedestal Fluctuations...'

    ## Process Data for each Part and Minute of a Night ##
    all_ped_fluct_data = []
    fluct_part = []
    fluct_minute = []

    #find unique parts :
    unique_parts =  list(sorted(set(fd_ped_data['part'])))

    # Select Part :
    for part in unique_parts:

        sel_part_fd_ped_data = fd_ped_data[fd_ped_data['part'] == part]

        unique_minutes =  list(sorted(set(sel_part_fd_ped_data['minute'])))

        #print unique_minutes

        # Select Unique Minutes :
        unique_minutes = unique_minutes[:-1] # exclude last minute

        for minute in unique_minutes:

            ped_fluct_data = _pedestal_fluctuation(sel_part_fd_ped_data, minute)

            fluct_part.append(part)
            fluct_minute.append(minute)

            all_ped_fluct_data.append(ped_fluct_data.tolist())

    # Number of Frames to be Generated given array of preprocessed data :
    nframes = len(fluct_part)

    ## Normalize PMT Pedestal Fluctuation with Logarithimic norm ##

    # create array of arrays of ped_fluct_norm from pedestal flucations :
    ped_fluct_norm = all_ped_fluct_data
    ped_fluct_norm[:] = np.array(ped_fluct_norm[:])
    ped_fluct_norm = np.array(ped_fluct_norm)

    # Normalize Pedestal Fluctations Using Night Max
    #norm = colors.LogNorm(vmax = np.amax(ped_fluct_norm), vmin = .1)
    #ped_fluct_norm = norm( ped_fluct_norm)

    # Normalize Pedestal Fluctations Using Frame Max
    for frame in range(0,nframes):
        norm = colors.LogNorm(vmax = np.amax(ped_fluct_norm[frame]), vmin = .1)
        ped_fluct_norm[frame] = norm( ped_fluct_norm[frame])

    # Convert Array of Arrays to List of List:
    ped_fluct_norm[:] = ped_fluct_norm[:].tolist()
    ped_fluct_norm = ped_fluct_norm.tolist()

    # Number of Frames to be Generated given array of preprocessed data :
    nframes = len(fluct_part)

    ## Save preprocessed Data in CSV Files ##

    print 'Saving Pre-Proccesed Data to CSV file...'

    # Convert to Pandas Data Frame :
    ped_fluct_df = pd.DataFrame(ped_fluct_data)
    ped_fluct_norm_df = pd.DataFrame(ped_fluct_norm)
    ped_fluct_info = {'part': fluct_part, 'minute': fluct_minute}
    ped_fluct_info_df = pd.DataFrame(ped_fluct_info)

    # Convert DataFrame to CSV File :
    ped_fluct_df.to_csv(OUT_DATA['fluct'].format(ymds))
    ped_fluct_norm_df.to_csv(OUT_DATA['fluct_norm'].format(ymds))
    ped_fluct_info_df.to_csv(OUT_DATA['fluct_info'].format(ymds))

    # Return Preproccesed Data
    return  all_ped_fluct_data, ped_fluct_norm, fluct_part, fluct_minute, nframes

def _pedestal_fluctuation(sel_part_fd_ped_data, minute):

    ped_i = sel_part_fd_ped_data[sel_part_fd_ped_data['minute'] == minute]
    ped_i = ped_i['pedestal']

    ped_f = sel_part_fd_ped_data[sel_part_fd_ped_data['minute'] == (minute + 1)]
    ped_f = ped_f['pedestal']

    # Convert to np arrays to manipulate :
    ped_i  = np.array(ped_i)
    ped_f  = np.array(ped_f)

    diff_ped_data = ped_f - ped_i

    # Make absolute Values :
    diff_ped_data = np.absolute(diff_ped_data)

    # Make all values positive by adding one to zeros
    diff_ped_data[diff_ped_data == 0] = diff_ped_data[diff_ped_data == 0] + 1

    return diff_ped_data

def _load_ped_data(frame):
    '''
    Load the Pedestal Data form HDF5 File containing pedestal fluctation, part and minute data.
    '''

    #print OUT_DATA['h5'].format(ymds)

    # Load HDF5 file :
    formatted_fd_ped_id = pd.read_hdf('test_id.h5','formatted_fd_ped_id',start=frame,stop=frame+1)

    part = formatted_fd_ped_id['part']
    minute = formatted_fd_ped_id['minute']

    # CSV File of pedestal fluctuations :
    formatted_fd_ped_data = pd.read_csv('test.csv',nrows=1,skiprows=frame)

    # Convert to list :
    formatted_fd_ped_data = formatted_fd_ped_data.values.tolist()
    formatted_fd_ped_data = formatted_fd_ped_data[0][:]

    #print formatted_fd_ped_data, part, minute

    return formatted_fd_ped_data, part, minute

def _load_pmt_positions():
    '''
    Load the pmt x and y tube positions for plotting hexagonal grid form numpy array row and col
    '''

    ## Create PMT Positions if none exists ##
    '''
    ## select a part and minute ##
    part, minute = 5, 0

    # Select data from part and minute :
    sel_part_fd_ped_data = fd_ped_data[fd_ped_data['part'] == part]
    sel_min_fd_ped_data = sel_part_fd_ped_data[sel_part_fd_ped_data['minute'] == minute]

    # Create HDF5 file of PMT Positions if not there :
    row, col = np.array(sel_min_fd_ped_data['row']), np.array(sel_min_fd_ped_data['col'])
    _create_pmt_positions(row, col)
    '''

    # Load HDF5 file :
    pmt_positions = pd.read_hdf('pmt_positions.h5','pmt_positions')

    pmt_x, pmt_y = np.array(pmt_positions['pmt_x']), np.array(pmt_positions['pmt_y'])

    return pmt_x, pmt_y

def _create_pmt_positions(row, col):
    '''
    Create the pmt x and y tube positions for plotting hexagonal grid form numpy array row and col.
    '''

    # Calculate PMT Position
    pmt_y = row * dy
    pmt_x = col * dx
    pmt_x[(row + 1) % 2 == 0] = pmt_x[(row + 1) % 2 == 0] + .5 * dx

    # Create Data Frame to save :
    pmt_positions = {'pmt_x': pmt_x, 'pmt_y': pmt_y}

    pmt_positions = pd.DataFrame(pmt_positions)

    # print pmt_position

    # Save to HDF5 fata file :
    pmt_positions.to_hdf('pmt_positions.h5','pmt_positions',format='table', mode='w')

def _dump_pedestal_dst(night, site, temp_dump_file):
    '''
    Dump FD pedestal data in .dst file to ascii file and clean up format
    '''

    y, m, d = YMD.findall(str(night))[0]

    print 'Dumping Pedestal DST Data for {0}...\r'.format( ymds )

    fd_pedestal_data =  FD_PED_DATA[site].format(y, m, d)

    # Dump Pedestal DST to temp ASCII File
    dump_cmd = 'dstdump {dst} > {tempfile} 2>/dev/null'.format(
        dst = fd_pedestal_data,
        tempfile = temp_dump_file
    )

    os.system(dump_cmd)

def _import_ped_txt(temp_dump_file):
    '''
    Import ascii file containing FD pedestal data into numpy array.
    '''
    print 'Reading Pedestal DST Data in...'

    # read dumped values into lists
    part = []            # Night Data Part
    minute = []            # Minute of Part
    mirror = []            # Mirror
    pmt = []            # PMT tube
    pedestal = []        # Pedestal Value

    f = open(temp_dump_file,'r')
    lines = f.readlines()
    for line in lines:
         if bool(re.search('  p',line)) == True :
            foo = line.split()
            if int(foo[6]) < 12 :
                part.append(int(foo[1]))
                minute.append(int(foo[3]))
                mirror.append(int(foo[6]))
                pmt.append(int(foo[8]))
                pedestal.append(float(foo[10]))
    f.close()

    mirror = np.array(mirror)
    pmt = np.array(pmt)

    #print mirror, pmt

    # Calculate PMT Row
    foo = mirror + 1
    foo = np.mod(foo,2)
    bar = np.mod(pmt,16)
    row = foo * 16 + bar
    #print col

    # Calculate PMT Column
    foo = mirror // 2
    bar = pmt // 16
    col = foo * 16 + bar

    # Create Structured Array of important Data Array to
    d = {'part': part,'minute': minute, 'mirror': mirror, 'pmt': pmt, 'pedestal': pedestal, 'row': row, 'col': col}

    fd_ped_data = pd.DataFrame(d)

    return fd_ped_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--night', type=int, help='8-digit night (yyyymmdd)')
    parser.add_argument('-s', '--site', type=int, choices=[0,1], help='site: 0 (BRM) or 1 (LR)')
    args = parser.parse_args()
    if args is not None:
        animate_fd_pedestal(args.night, args.site)

    else:
        logging.error('Missing required arguments')

#---# End of Script #---#
