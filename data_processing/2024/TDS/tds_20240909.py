import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
import os
import json
import numpy as np
from scipy.optimize import least_squares, OptimizeResult

over_sqrt_pi = 1. / np.sqrt(2. * np.pi)


def gaussian(x, mu, sigma):
    p = over_sqrt_pi / sigma
    arg = -0.5 * np.power((x - mu) / sigma)
    return p * np.exp(arg)

def sum_gaussians(x, b):
    n = len(x)
    m = len(b)
    y = np.zeros(n, dtype=float)
    param_ord = np.arange(0, m)
    
