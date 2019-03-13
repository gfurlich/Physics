#-------------------------------------------------------------------------------------#
#
#    File :                 fd_pedestal_animation_good_qual.py
#    Author :               Greg Furlich
#    Date Created :         2018-07-12
#
#    Purpose: To animate the pedestal values of BR and LR to see the night sky video. Based on fd_pedestal_animation.py but optimized for hex pixel and capture pedestal flucuations by taking differenceof pedestal values between subsequent minutes
#
#
#    Execution :   python fd_pedestal_animation_good_qual.py -n <YYYYMCDD> -s <site>
#    Example :     python fd_pedestal_animation_good_qual.py -n 20170225 -s 0
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

parent = os.environ['taresearch']+'/FD_Ped_Weather/'

FD_PED_DATA = {
    #0: '/ssd32g/rtdata/calibration/fdped/black-rock/y{0}m{1}d{2}.ped.dst.gz',
    #1: '/ssd32g/rtdata/calibration/fdped/long-ridge/y{0}m{1}d{2}.ped.dst.gz',
    0: parent + 'fdped/black-rock/y{0}m{1}d{2}.ped.dst.gz',
    1: parent + 'fdped/long-ridge/y{0}m{1}d{2}.ped.dst.gz',
}


OUT_DIR = {
    'frame': parent + 'Good_GIF/{0}_dh/',
    'gif': parent + 'Good_GIF/',
}

IN_DATA = {
    'frame':   parent + 'Data/{0}_ped_fluct.h5',
}

OUT_FIG = {
    'frame': 'frame_p{:02d}_min{:02d}.png',
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

## Approximate Field of View Length  ##

lr_x_angle_min = 18     # CW from North
lr_x_angle_max = 125    # CW from North

br_x_angle_min = 244    # CW from North
br_x_angle_max = 350    # CW from North

x_angle_range = 108

y_angle_max = 33        # degrees from horizon
y_angle_min = 3         # degrees from horizon
y_angle_range = y_angle_max - y_angle_min

y_10 = ( 10 - y_angle_min ) * (nrows * dy ) / y_angle_range
y_20 = ( 20 - y_angle_min ) * (nrows * dy ) / y_angle_range
y_30 = ( 30 - y_angle_min ) * (nrows* dy ) / y_angle_range
y_34 = ( 34 - y_angle_min ) * (nrows* dy ) / y_angle_range

# BR FOV
# NW
br_315 = ( 315 - br_x_angle_min ) * (ncols * dx ) / x_angle_range
# W
br_270 = ( 270 - br_x_angle_min ) * (ncols * dx ) / x_angle_range
#
br_240 = ( 240 - br_x_angle_min ) * (ncols * dx ) / x_angle_range
br_300 = ( 300 - br_x_angle_min ) * (ncols * dx ) / x_angle_range
br_330 = ( 330 - br_x_angle_min ) * (ncols * dx ) / x_angle_range
br_255 = ( 255 - br_x_angle_min ) * (ncols * dx ) / x_angle_range
br_285 = ( 285 - br_x_angle_min ) * (ncols * dx ) / x_angle_range
br_345 = ( 345 - br_x_angle_min ) * (ncols * dx ) / x_angle_range

# LR FOV
# E
lr_90 = ( 90 - lr_x_angle_min ) * (ncols * dx ) / x_angle_range
# NE
lr_45 = ( 45 - lr_x_angle_min ) * (ncols * dx ) / x_angle_range
#
lr_30 = ( 30 - lr_x_angle_min ) * (ncols * dx ) / x_angle_range
lr_60 = ( 60 - lr_x_angle_min ) * (ncols * dx ) / x_angle_range
lr_90 = ( 90 - lr_x_angle_min ) * (ncols * dx ) / x_angle_range
lr_120 = ( 120 - lr_x_angle_min ) * (ncols * dx ) / x_angle_range
lr_75 = ( 75 - lr_x_angle_min ) * (ncols * dx ) / x_angle_range
lr_105 = ( 105 - lr_x_angle_min ) * (ncols * dx ) / x_angle_range

## Initialize Frame Variables ##
night_sky = plt.figure()

cmap = cm.inferno

def animate_fd_pedestal(night, site):

    # Data Selection :
    global ymds, ymd, y, m, d, s
    y, m, d = YMD.findall(str(night))[0]
    ymd = str(y)+str(m)+str(d)
    ymds = 'y{0}m{1}d{2}s{3}'.format(y, m, d, site)
    s = site

    print 'Animating Pedestal Fluctuations for {0}'.format(ymds)

    # Import Pedestal Frame data :
    ped_fluct_df, ped_fluct_norm_df, frame_info_df = _load_ped_data()

    # Check if outfile exists:
    out_dir = OUT_DIR['frame'].format(ymds)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Create animation :
    _create_pedestal_animation(ped_fluct_df, ped_fluct_norm_df, frame_info_df)

def _create_pedestal_animation(ped_fluct_df, ped_fluct_norm_df, frame_info_df):

    nframes = len(frame_info_df)

    print 'Generating {0} Frames'.format(nframes)
    ## Define Global Variables for Frame Creation ##

    # Load PMT Position :
    global g_pmt_x, g_pmt_y
    g_pmt_x, g_pmt_y = _load_pmt_positions()
    # print g_pmt_x, g_pmt_y

    global g_all_ped_fluct_data, g_ped_fluct_norm, g_frame_part, g_frame_minute, g_frame_max, g_frame_min, g_frame_mean, g_frame_sigma, g_frame_time

    ## Create Frames ##

    # Select Frame :
    for frame in range(0,nframes):

        # print 'Rendering Pedestal Sky View for frame: {:02d} / {:02d}\r'.format(int(frame), int(nframes))

        g_all_ped_fluct_data =  ped_fluct_df.loc[frame]
        g_ped_fluct_norm = ped_fluct_norm_df.loc[frame]
        g_frame_part, g_frame_minute = frame_info_df['frame_part'].loc[frame], frame_info_df['frame_minute'].loc[frame]
        g_frame_max, g_frame_min, g_frame_mean, g_frame_sigma = frame_info_df['frame_max'].loc[frame], frame_info_df['frame_min'].loc[frame], frame_info_df['frame_mean'].loc[frame], frame_info_df['frame_sigma'].loc[frame]
        g_frame_time = frame_info_df['frame_time'].loc[frame]

        # print g_frame_part, g_frame_minute, g_frame_max, g_frame_min, g_frame_mean, g_frame_sigma
        # print g_all_ped_fluct_data, g_ped_fluct_norm

        # Render Frame with desired Pixel Type :
        __create_frame_diff_hex_pixel(frame)

    ## Create GIF of Night From Frames using ImageMagick ##
    out_gif = OUT_DIR['gif'] + OUT_FIG['gif'].format(ymds)
    out_dir = OUT_DIR['frame'].format(ymds)

    # print 'Rendering Pedestal Fluctuation GIF : '+out_gif

    # Use ImageMagick and System commands:
    #os.system('convert -delay 5 -loop 0 -layers Optimize'+out_dir+'*.jpg '+out_gif)
    # os.system('convert '+out_dir+'*.jpg '+out_gif)

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
    #ax.set_axis_bgcolor('black')
    ax.set_facecolor('black')

    # Frame Limits :
    plt.xlim([-r, ncols*dx])        # X Range
    plt.ylim([-r, (nrows)*dy])        # Y Range

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
    ax.scatter(g_pmt_x, g_pmt_y, s=(r * 4.1)**2, c = cmap(g_ped_fluct_norm), marker = 'h', edgecolor='none')

    # Add text information to plot :
    plt.text(0,-10*r,r'{0} UTC '.format(g_frame_time.strftime("%Y-%m-%d %H:%M:%S")),color='white', fontsize=10)
    if s == 0 :         # BR
        plt.text(ncols*dx*.66,-10*r,r'Black Rock Field of View',color='white', fontsize=8)
    elif s == 1 :       # LR
        plt.text(ncols*dx*.66,-10*r,r'Long Ridge Field of View',color='white', fontsize=8)

    # plt.text(2*r,-6*r,r'part: {0} $\Delta$ min: {1} - {2} '.format(int(g_frame_part),int(g_frame_minute),int(g_frame_minute + 1)),color='white', fontsize=14, weight='bold')
    # plt.text(2*r,-6*r,r'part: {0} $\Delta$ min: {1} - {2} max={3:.0f} mean={4:.0f} std={5:.0f}'.format(int(g_frame_part),int(g_frame_minute),int(g_frame_minute + 1), g_frame_max, g_frame_mean, g_frame_sigma),color='white', fontsize=12, weight='bold')

    ## Add Angle Markers to Plot ##

    ## Y angle markers ##

    # Markers :
    plt.plot([-r, 0], [y_10, y_10], 'w-', linewidth=.85)
    plt.plot([-r, 0], [y_20, y_20], 'w-', linewidth=.85)
    plt.plot([-r, 0], [y_30, y_30], 'w-', linewidth=.85)

    plt.plot([ncols*dx - r, ncols*dx], [y_10, y_10], 'w-', linewidth=.85)
    plt.plot([ncols*dx - r, ncols*dx], [y_20, y_20], 'w-', linewidth=.85)
    plt.plot([ncols*dx - r, ncols*dx], [y_30, y_30], 'w-', linewidth=.85)

    # Marker Labels
    plt.text(-5 * r, y_10 - 1 * r, r'{0}$^\circ$ '.format(10),color='white', fontsize=4, weight='bold')
    plt.text(-5 * r, y_20 - 1 * r, r'{0}$^\circ$ '.format(20),color='white', fontsize=4, weight='bold')
    plt.text(-5 * r, y_30 - 1 * r, r'{0}$^\circ$ '.format(30),color='white', fontsize=4, weight='bold')
    plt.text(-5 * r, y_34 - 1 * r, r'[ Elev. $^\circ$ ]'.format(330),color='white', fontsize=3, weight='bold')

    ## X markers ##
    if s == 0 :         # BR

        # Markers :
        plt.plot([br_270, br_270], [-r, 0], 'w-', linewidth=.85)
        plt.plot([br_315, br_315], [-r, 0], 'w-', linewidth=.85)
        plt.plot([br_300, br_300], [-r, 0], 'w-', linewidth=.85)
        plt.plot([br_330, br_330], [-r, 0], 'w-', linewidth=.85)
        plt.plot([br_255, br_255], [-r, 0], 'w-', linewidth=.85)
        plt.plot([br_285, br_285], [-r, 0], 'w-', linewidth=.85)
        plt.plot([br_345, br_345], [-r, 0], 'w-', linewidth=.85)

        plt.plot([br_270, br_270], [nrows*dy - r , nrows*dy], 'w-', linewidth=.85)
        plt.plot([br_315, br_315], [nrows*dy - r , nrows*dy], 'w-', linewidth=.85)
        plt.plot([br_300, br_300], [nrows*dy - r , nrows*dy], 'w-', linewidth=.85)
        plt.plot([br_330, br_330], [nrows*dy - r , nrows*dy], 'w-', linewidth=.85)
        plt.plot([br_255, br_255], [nrows*dy - r , nrows*dy], 'w-', linewidth=.85)
        plt.plot([br_285, br_285], [nrows*dy - r , nrows*dy], 'w-', linewidth=.85)
        plt.plot([br_345, br_345], [nrows*dy - r , nrows*dy], 'w-', linewidth=.85)

        # Marker Labels
        plt.text(br_270 - 4 * dx, -4*r, r'W ({0}$^\circ$) '.format(270),color='white', fontsize=6, weight='bold')
        plt.text(br_315 - 4 * dx, -4*r, r'NW ({0}$^\circ$) '.format(315),color='white', fontsize=6, weight='bold')
        plt.text(br_300 - 1 * dx, -4*r, r'{0}$^\circ$ '.format(300),color='white', fontsize=4, weight='bold')
        plt.text(br_330 - 1 * dx, -4*r, r'{0}$^\circ$ '.format(330),color='white', fontsize=4, weight='bold')
        plt.text(br_255 - 1 * dx, -4*r, r'{0}$^\circ$ '.format(255),color='white', fontsize=4, weight='bold')
        plt.text(br_285 - 1 * dx, -4*r, r'{0}$^\circ$ '.format(285),color='white', fontsize=4, weight='bold')
        plt.text(br_345 - 1 * dx, -4*r, r'{0}$^\circ$ '.format(345),color='white', fontsize=4, weight='bold')
        plt.text(0, -3.5*r, r'[ $^\circ$ CW from N ] '.format(330),color='white', fontsize=3, weight='bold')

    elif s == 1 :       # LR

        # Markers :
        plt.plot([lr_90, lr_90], [-r, 0], 'w-', linewidth=.85)
        plt.plot([lr_45, lr_45], [-r, 0], 'w-', linewidth=.85)
        plt.plot([lr_30, lr_30], [-r, 0], 'w-', linewidth=.85)
        plt.plot([lr_60, lr_60], [-r, 0], 'w-', linewidth=.85)
        plt.plot([lr_120, lr_120], [-r, 0], 'w-', linewidth=.85)
        plt.plot([lr_75, lr_75], [-r, 0], 'w-', linewidth=.85)
        plt.plot([lr_105, lr_105], [-r, 0], 'w-', linewidth=.85)

        plt.plot([lr_90, lr_90], [nrows*dy - r, nrows*dy], 'w-', linewidth=.85)
        plt.plot([lr_45, lr_45], [nrows*dy - r, nrows*dy], 'w-', linewidth=.85)
        plt.plot([lr_30, lr_30], [nrows*dy - r, nrows*dy], 'w-', linewidth=.85)
        plt.plot([lr_60, lr_60], [nrows*dy - r, nrows*dy], 'w-', linewidth=.85)
        plt.plot([lr_120, lr_120], [nrows*dy - r, nrows*dy], 'w-', linewidth=.85)
        plt.plot([lr_75, lr_75], [nrows*dy - r, nrows*dy], 'w-', linewidth=.85)
        plt.plot([lr_105, lr_105], [nrows*dy - r, nrows*dy], 'w-', linewidth=.85)

        # Marker Labels
        plt.text(lr_90 - 2 * dx, -4*r, r'E ({0}$^\circ$) '.format(90),color='white', fontsize=6, weight='bold')
        plt.text(lr_45 - 2 * dx, -4*r, r'NE ({0}$^\circ$) '.format(45),color='white', fontsize=6, weight='bold')
        plt.text(lr_30 - 1 * dx, -4*r, r'{0}$^\circ$ '.format(30),color='white', fontsize=4, weight='bold')
        plt.text(lr_60 - 1 * dx, -4*r, r'{0}$^\circ$ '.format(60),color='white', fontsize=4, weight='bold')
        plt.text(lr_120 - 1 * dx, -4*r, r'{0}$^\circ$ '.format(120),color='white', fontsize=4, weight='bold')
        plt.text(lr_75 - 1 * dx, -4*r, r'{0}$^\circ$ '.format(75),color='white', fontsize=4, weight='bold')
        plt.text(lr_105 - 1 * dx, -4*r, r'{0}$^\circ$ '.format(105),color='white', fontsize=4, weight='bold')
        plt.text(0, -3.5*r, r'[ $^\circ$ CW from N ] '.format(330),color='white', fontsize=3, weight='bold')

    # Save Figure Name :
    out_fig = OUT_DIR['frame'].format(ymds) + OUT_FIG['frame'].format(int(g_frame_part),int(g_frame_minute))

    # Save Plot :
    #night_sky.savefig(out_fig, dpi = 200, bbox_inches='tight', facecolor = 'black', pad_inches=-.01)
    night_sky.savefig(out_fig, dpi = 200, bbox_inches='tight', facecolor = 'black')

    night_sky.clf()

def _load_ped_data():
    '''
    Load the Pedestal Data form HDF5 File containing pedestal fluctation, part and minute data.
    '''

    # Load HDF5 File :
    store_df = pd.HDFStore(IN_DATA['frame'].format(ymds))

    # Save DataFrame to HDF5 file :
    ped_fluct_df = store_df.get('ped_fluct_df')
    ped_fluct_norm_df = store_df.get('ped_fluct_norm_df')
    frame_info_df = store_df.get('frame_info_df')

    # Close HDF5 File :
    store_df.close()

    return ped_fluct_df, ped_fluct_norm_df, frame_info_df

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
    load_pmt_positions_file = pd.HDFStore('pmt_positions.h5')
    pmt_positions = load_pmt_positions_file.get('pmt_positions')
    load_pmt_positions_file .close()

    pmt_x, pmt_y = pmt_positions['pmt_x'].tolist(), pmt_positions['pmt_y'].tolist()

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

    pmt_positions = pd.DataFrame(pmt_positions, dtype=np.float32)

    print pmt_position

    # HDF5 File :
    store_pmt_positions = pd.HDFStore('pmt_positions.h5')

    # Save DataFrame to HDF5 file :
    store_pmt_positions['pmt_positions'] = pmt_positions

    store_pmt_positions.close()

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
