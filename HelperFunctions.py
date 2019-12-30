'''
	A set of helper functions which will be useful for 
	this study.
'''

import os
import pandas as pd
import numpy as np
import tarfile
import matplotlib.pyplot as plt

# Helper function for saving figures efficiently 
def save_fig(fig_name, fig_dir = './Figures', tight_layout=True, fig_extension="png", resolution=300):
    if not os.path.exists(fig_dir): 
	os.makedirs(fig_dir)
    path = os.path.join(fig_dir, fig_name + "." + fig_extension)
    print("Saving figure", fig_name)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Function to fetch the data from a tgz 
def fetch_housing_data():
    tgz_path = 'housing.tgz'
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall()
    housing_tgz.close()

# Function to load the data with Pandas
def load_housing_data():
    csv_path = 'housing.csv'
    return pd.read_csv(csv_path)

# An example of how one might split the testing and training sets. Note
# this this function has no random state, so would give a different testing
# and training set each time. Not recommended. 
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data)) # Shuffle the data first
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size] # Sample upto the testing data size
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices] # Return the located indices of the training and testing sets
