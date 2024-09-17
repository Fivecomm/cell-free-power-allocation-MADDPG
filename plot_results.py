#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
This Python script plor results from research work:

Guillermo García-Barrios, Manuel Fuentes, David Martín-Sacristán, "A Novel 
MADDPG Algorithm for Efficient Power Allocation in Cell-Free Massive MIMO," 
IEEE Wireless Communications and Networking Conference (WCNC), Milan, Italy, 
2025. [Pending acceptance]

This is version 1.0 (Last edited: 2024-09-17)

@author: Guillermo Garcia-Barrios

License: This code is licensed under the GPLv2 license. If you in any way
use this code for research that results in publications, please cite our
paper as described in the README file.
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt

##############################################################################
## PARAMETERS TO SET UP

# Cell-free scenario
TYPE_UES = 'movingUEs'
SCENARIO = 'scenario_01'
# Power allocation strategy
PA_STRATEGY = 'sumSE'
# Results path
RESULTS_PATH = 'results/'


##############################################################################
## FUNCTIONS

def cdf(data):
    """
    Compute the Cumulative Distribution Function (CDF) of the input data.

    Parameters:
    data (np.array): The input data as a numpy array.

    Returns:
    sorted_data (np.array): The input data sorted in ascending order.
    p (np.array): The values of the CDF for the corresponding elements in 
    sorted_data.
    """

    # Sort the data in ascending order
    sorted_data = np.sort(data, axis=0)

    # Calculate the proportional values of samples. This is done by creating an
    # array of indices (from 0 to len(data)-1), dividing each index by 
    # (len(data) - 1), and scaling by 1.0 to ensure the result is a floating 
    # point number. This gives us the percentile ranks of the data which we 
    # will use as the CDF values.
    p = 1. * np.arange(len(data)) / (len(data) - 1)

    # Return the sorted data and the corresponding CDF values
    return sorted_data, p


##############################################################################
## LOAD DATA
load_path = RESULTS_PATH + TYPE_UES + '/' + SCENARIO + '/'
SE_test = np.load(load_path + 'SE_' + PA_STRATEGY + '_test.npy')
SE_maddpg = np.load(load_path + 'SE_' + PA_STRATEGY + '_predicted.npy')
rhokl_test = np.load(load_path + 'rhokl_' + PA_STRATEGY + '_test.npy')
rhokl_maddpg = np.load(load_path + 'rhokl_' + PA_STRATEGY + '_predicted.npy')

nbr_setups = len(SE_test[0])


##############################################################################
## PLOT CDF 

# Reshape 2D matrix SE into a vector
SE_test = np.reshape(SE_test, (-1, 1)) 
SE_maddpg = np.reshape(SE_maddpg, (-1, 1))    

# Calculate CDF
sorted_SE_test, cdf_test = cdf(SE_test)
sorted_SE_maddpg, cdf_maddpg = cdf(SE_maddpg)

# Plot CDFs
lw = 3
plt.figure(figsize=(8, 7))
plt.plot(sorted_SE_test, cdf_test,
         color='k', linestyle='solid', linewidth=lw, label='sumSE')
plt.plot(sorted_SE_maddpg, cdf_maddpg,
         color='b', linestyle='dashed', linewidth=lw, label='MADDPG')
plt.legend(loc='lower right')
plt.xlabel('SE por UE (bit/s/Hz)')
plt.ylabel('CDF')
plt.xlim(0,18)
plt.grid(True)
plt.tight_layout()
plt.show()