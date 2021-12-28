import numpy as np
import matplotlib.pyplot as plt
import psrsigsim as pss
from numba import jit
from scipy.stats import skewnorm


@jit(nopython=True)
def gaussian(x, mu, sig):
    """
    Compute 1-D Guassian 
    Parameters
    ----------
    x : point value [1d]
    mu : mean value 
    sig : deviation  
    Returns
    -------
    Distance : 
    guassian evaluated at x 
    """
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

