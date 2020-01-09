import pandas as pd
import numpy as np
import os as os
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import json
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

historical_data_dir = "./data/historical/pickles"
class OptionStatsViewer:

    def __init__(self, data_file_name):
        self.df = OptionStatsViewer.read_data_file(data_file_name)

    @staticmethod
    def read_data_file(filename):
        data_file = historical_data_dir + os.path.sep + filename
        if not os.path.exists(data_file):
            print("Error, historical data does not exist")
        df = pd.read_pickle(data_file)
        return df

