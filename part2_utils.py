import copy
import torch
import torch.nn as nn
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random

from part2_house_value_regression import *


def init_weights(m):
    """
    Initializes the weights using Xavier uniform method and biases using uniform distribution from 0 to 1.
    Args:
        m: network to be initialized

    Returns:
        None
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.uniform_(m.bias)


def set_seeds(seed):
    """
    Sets seed for numpy and PyTorch random generators.

    Args:
        seed {int}: seed for the random generators

    Returns:
        None
    """
    torch.manual_seed(seed)
    np.random.seed(seed)


def custom_ohe(oc_prox, import_dict=None, number_of_columns=5):
    """
    Function that takes ocean proximity column and converts to one hot encoded dataframe. Custom one hot encoding was selected to enable handling nan values more robustly and keep track of columns.
    Args:
        oc_prox -  Ocean Proximity Value

    Returns:
        - convert_to_df {type(DataFrame)} - converted to DataFrame
        - unique_dict {dict} - unique columns
    """
    # Converts the final column of median house value into a numpy array for processing and also find unique values in dataframe.
    ocean_prox_arr = np.array(oc_prox)
    unique_entries = oc_prox.unique()
    unique_dict = {}

    # Make dictionary of keys and values of unique entries
    nan_flag = 0
    for iter, entry in enumerate(unique_entries):
        if str(entry) != "nan" and nan_flag == 0:
            unique_dict[str(entry)] = iter
        elif nan_flag == 1 and str(entry) != "nan":
            unique_dict[str(entry)] = iter - 1  # process the presence of nans
        else:
            nan_flag = 1  # case where nan has been flagged by the iter key

    if import_dict is not None:
        unique_dict = import_dict
    make_vals = []
    # Loop through Ocean Proximity Array and Make a 2D Arr of Values
    for val in ocean_prox_arr:
        tmp_arr = [0 for i in range(number_of_columns)]
        str_val = str(val)
        if str_val in unique_dict and unique_dict[str_val] < number_of_columns:
            tmp_arr[unique_dict[str_val]] = 1
        else:
            tmp_arr = [0.5 for i in range(number_of_columns)]
        make_vals.append(tmp_arr)

    # Converts array to a dataframe with corresponding one hot encoded values
    convert_to_df = pd.DataFrame(make_vals)
    return convert_to_df, unique_dict


def plot_metrics(train_losses, val_losses):
    """
    Plots the training and validations losses.

    Args:
        train_losses: training losses
        val_losses: validation losses

    Returns:
        None
    """
    x = range(len(val_losses))

    # plot lines
    plt.plot(x, val_losses, label="val losses")
    plt.plot(x, train_losses, label="train losses")
    plt.legend()
    plt.show()

    return


def batch_data(x, y, batch_size=32):
    """
    Separates input features and outputs to batches of the specified size and ignores records that would overflow the batch size.
    Args:
        x: input features
        y: outputs
        batch_size {int}: batch size

    Returns:
        - {numpy.ndarray} -- Input features splits to batches
        - {numpy.ndarray} -- Outputs split to batches
    """
    x_len = len(x)
    if x_len < batch_size:
        return np.array(x), np.array(y)

    # Check if partially filled batches are to be discarded
    overflow = len(x) % batch_size

    # Check if overflow needs to be trimmed
    if overflow != 0:
        x = x[: -overflow]
        y = y[: -overflow]

    # Reshape training data into batches
    x = np.reshape(x, (x_len // batch_size, batch_size, x.shape[1]))
    y = np.reshape(y, (x_len // batch_size, batch_size, y.shape[1]))

    return x, y


def partition_inputs(x, y, validation_split=0.1, test_split=0.1):
    """
    Partitions inputs and outputs into a training, validation and test sets.

    Args:
        - x {numpy.ndarray}:
        - y {numpy.ndarray}:
        - validation_split {float}: ratio for the validation set - has to be between 0 and 1
        - test_split {float}: ratio for the validation set - has to be between 0 and 1

    Returns:
        - {numpy.ndarray} - inputs for the training set
        - {numpy.ndarray} - outputs for the training set
        - {numpy.ndarray} - inputs for the validation set
        - {numpy.ndarray} - outputs for the validation set
        - {numpy.ndarray} - inputs for the test set
        - {numpy.ndarray} - outputs for the test set
    """
    assert len(x) == len(y)

    # Split off testing and validation
    x_len = len(x)

    # Set generation
    x_test = x[int((1 - test_split) * x_len):]
    y_test = y[int((1 - test_split) * x_len):]

    x_val = x[int((1 - test_split - validation_split) * x_len): int((1 - test_split) * x_len)]
    y_val = y[int((1 - test_split - validation_split) * x_len): int((1 - test_split) * x_len)]

    x_train = x[: int((1 - test_split - validation_split) * x_len)]
    y_train = y[: int((1 - test_split - validation_split) * x_len)]

    return x_train, y_train, x_val, y_val, x_test, y_test
