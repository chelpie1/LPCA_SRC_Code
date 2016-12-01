# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 15:49:48 2016
@author: Chelsea Weaver

CW note: To run from terminal (e.g., iPython) first type 
         "From Normalize import Normalize")
"""

import numpy as np
from numpy import *

def Normalize(data):
    """ Normalize(data) normalizes the matrix "data" to have columns with 
    norm 1.
   
    """
    [m,n] = np.shape(data)
    col_norms = np.zeros((n,1))
    for i in range(n):
        col_norms[i] = np.linalg.norm(data[:,i])
        norm_copies = tile(col_norms.T,(m,1))

    data_norm = data / norm_copies
    pos_of_nans = np.isnan(data_norm)
    data_norm[pos_of_nans] = 0
    
    return data_norm



