# Research

#### File : Research/readme.md
#### Author : Greg Furlich

Comment : This is the git repository of selected projects made for my research in Cosmic Rays with the Telescope Array Project at the University of Utah, Department of Physics and Astronomy, Institute of High Energy Astrophysics.

## Contents ##

1) **atmosphere.cpp** : C++ and Root defined functions to calculate the matter traversed through when traveling through the atmosphere. Used to determine atmospheric interaction depth for a cosmic ray event.

2) **fd_pedestal_animation_good_qual.py** : Generate animations of the pedestal values of our arrangement of Photomultiplier Tubes (PMTS) for our Cosmic Ray Detector Sites to make animations of the night sky in the field of view to help determine the weather during observations.

<p align="center">
    <img src="https://github.com/gfurlich/Research/blob/master/GIFs/clear.gif">
    <b>Example Animation of PMT Pedestals Clear Night</b>
    <br>
    <img src="https://github.com/gfurlich/Research/blob/master/GIFs/cloudy.gif">
    <b>Example Animation of PMT Pedestals Cloudy Night</b>
</p>

3) **TAMap/TA_topo.ipynb** and **TAMap/TA_sate.ipynb** : Jupyter notebooks for remote image processing for overlaying features of the Telescope Array Cosmic Ray Observatory over Landsat 8 images for a satellite image and STRM data for a topo and relief map.

<p align="center">
    <img src="https://github.com/gfurlich/Research/blob/master/TAMap/ta_map.svg">
    <b>Telescope Array Cosmic Ray Observatory Satellite Map</b>
    <img src="https://github.com/gfurlich/Research/blob/master/TAMap/ta_topo.svg">
    <b>Telescope Array Cosmic Ray Observatory Topo Map</b>
    <img src="https://github.com/gfurlich/Research/blob/master/TAMap/ta_relief.svg">
    <b>Telescope Array Cosmic Ray Observatory Relief Map</b>
</p>

4) **fd_pedestal_rnn_vectorization_v_chpc.py** : To load in all fd pedestal preprocessed data stored in Pandas DataFrame into Numpy array 3D array and pad Frames so they all have the same length for use in a Recurrent Convolution Neural Network (RCNN).
