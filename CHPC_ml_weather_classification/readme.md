# CHPC

#### File : Research/CHPC_ml_weather_classification/readme.md
#### Author : Greg Furlich

Comment : Machine learning Dense Neural Network (DNN), Convolution Neural Network (CNN), Recurrent Neural Network (RNN), and Recurrent Convolution Neural Network (RCNN) models run on the [Center for High Performance Computing (CHPC)](https://chpc.utah.edu/) computational clusters at the University of Utah. The RCNN was ultimately used to classify the weather in the detector's field of view using animations of false color images from each minute of operation created from the PMT pedestal values and their nominal pointing direction. Given the size of the data, issues such as padding the animation arrays to be equal shape and using a data generator to load in each batch for training were addressed.
