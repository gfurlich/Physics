# Research

### File : Research/readme.md
### Author : Greg Furlich

Comment : Portfolio of selected projects made for my research in Cosmic Rays with the Telescope Array Project at the University of Utah, Department of Physics and Astronomy, Institute of High Energy Astrophysics.

## Contents ##

**atmosphere.cpp** : C++ and Root defined functions to calculate the matter traversed through when traveling through the atmosphere. Used to determine atmospheric interaction depth for a cosmic ray event.

**fd_pedestal_animation.py** : Generate images and animations used for classifying the training set by eye used for training the neural networks designed in **CHPC/**. These animation pixel values were taken from the pedestal values of our arrangement of Photomultiplier Tubes (PMTS) for the Telescope Array Cosmic Ray fluorescence detector Black Rock and Long Ridge sites.

**fd_pedestal_animation_good_qual.py** : Generate animations of the pedestal values of our arrangement of Photomultiplier Tubes (PMTS) for the Telescope Array Cosmic Ray fluorescence detector Black Rock and Long Ridge sites. These animations display the night sky in the field of view to help determine the weather when the fluorescence detectors were acquiring data. Examples of these animations can be [viewed on my website](https://gregfurlich.com/posts/telescope-array-machine-learing-weather-classification.html).

**fd_pedestal_rnn_vectorization_v_chpc.py** : Load in all FD pedestal preprocessed data stored in Pandas DataFrame into a NumPy 3D array and pad array with zeroed Frames in front so they all have the same length for use in a Recurrent Convolution Neural Network (RCNN).

**CHPC/** : Contains all machine learning codes building up from a Deep Neural Network (DNN), to a Convolution Neural Network (CNN), then a Recurrent Neural Network (RNN), and finally combing the RNN and CNN into a RCNN in order classify temporal progressions snapshots of the night sky into weather classes over a time period. Contains slurm (`.slm`) scripts for queuing and calling python scripts which use the Keras modules to train and classify the sequences of snapshots. Some scripts utilize a data generator for loading in and padding these 3D arrays and training the model and utilize CPU nodes at the Center for High Performance Computing (CHPC) at the University of Utah.

**TAMap/TA_topo.ipynb** and **TAMap/TA_sate.ipynb** : Jupyter notebooks for remote image processing for overlaying features of the Telescope Array Cosmic Ray Observatory over Landsat 8 images for a satellite image and  Shuttle Radio Topography Mission () data for a topo and shaded relief map.

<p align="center">
    <img src="https://github.com/gfurlich/Research/blob/master/TAMap/ta_map.png">
    <b>Telescope Array Cosmic Ray Observatory Satellite Map</b>
    <img src="https://github.com/gfurlich/Research/blob/master/TAMap/ta_topo.png">
    <b>Telescope Array Cosmic Ray Observatory Topo Map</b>
    <img src="https://github.com/gfurlich/Research/blob/master/TAMap/ta_relief.png">
    <b>Telescope Array Cosmic Ray Observatory Relief Map</b>
</p>
