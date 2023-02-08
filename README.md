
# Introduction to Machine Learning Coursework 2: Artificial Neural Networks

This repository contains the implementation to coursework 2 of COMP70050 to estimate housing prices from given data. Please find more information and instructions below. 

## Acknowledgments
I definitely didn't complete this project on my own and dedicate my thanks to the other who spent much time working with me on this brilliant project (contributors are all listed below). The following repository contains the pickle file of our neural network and describes our approach for conducting hyperparameter sweeping to find the best network architecture. It was a gruelling but fantastic project experience and we learnt a lot about neural networks, hyperparameter sweeping and data preprocesing from implementing this project.   

## Contributors
- Matthew Setiawan (ms3120)
- Michal Palic (mp3120)
- Vaclav Pavlicek (vp920)
- James Ong (jjo20)

## Overview
The main code was written in `part1_nn_lib.p` and `part2_house_value_regression.py`. Below each file is described in more detail.

### Part 1
Part 1 contains self defined classes for different activation functions, preprocessor, testing function, and training function. The full list of classes is given below:

- **xavier_init** - Used to return random, uniformly distributed weights
- **Layer** - defines the abstract layer class
- **MSELossLayer** - Computes mean-squared error between y_pred and y_target
- **CrossEntropyLossLayer**  - Computes the softmax followed by the negative log-likelihood loss
- **SigmoidLayer** - Applies sigmoid function elementwise
- **ReluLayer** - Applies Relu function elementwise
- **LinearLayer** - Performs affine transformation of input
- **MultiLayerNetwork** - A network consisting of stacked linear layers and activation functions
- **save_network** - Utility function to pickle `network` at file path `fpath`
- **load_network** - Utility function to load network found at file path `fpath`
- **Trainer** - Object that manages the training of a neural network
- **Preprocessor** - Object used to apply "preprocessing" operation to datasets
- **main** - Main function used to use to other objects and functions

Part 1 can be run with the `python3 part1_nn_lib.p` command.

### Part 2
Part 2 contains a regressor which is able to initialize a network, preprocess data, train the network with data, predict output with a forward pass, and obtain a MSE loss value, it also contains a hyper parameter searching function to apply a grid search to find best parameters and utility functions such as saving and loading previously created networks. A more detailed list of classes is listed below:

- **Regressor** - Contains constructor for neural network, preprocessor, fitter, score function and predict function.
- **save_regressor** - Utility function to pickle `network` at file path `fpath`
- **load_regressor** - Utility function to load network found at file path `fpath`
- **RegressorHyperParameterSearch**  - Performs a grid search to find optimum parameters on network
- **main** - Main function used to use to other objects and functions

Part 2 can be run with the `python3 part2_house_value_regression.py` command.

A couple of example main functions were provided in `part2_house_value_regression.py` to .
- `main_load_regressor_and_predict_whole_dataset()` - loads the regressor from `part2_model.pickle` and runs the prediction of `housing.csv` on it 
- `copy_network_weights_and_biases()` - copies networks and weight from one Regressor to another - saves time on retraini
- `RegressorHyperParameterSearchSeed` - runs the algorithm for finding the best hyperparameters
- `example_main` - example main provided in the code skeleton

## Other Files

### housing.csv
This file contains the raw data that the neural network was trained on.
