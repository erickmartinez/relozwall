import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Tuple

f_pattern = re.compile(r'(.*)?\:\s+(.*)')

def LastNlines(fname: str, N: int) -> str:
    """
    Function to read
    last N lines of the file

    Parameters
    ----------
    fname: str
        The path to the file
    N: the number of lines from the end of the file

    Returns
    -------
    str:
        The last N lines in the file
    """
    # assert statement check
    # a condition
    assert N >= 0

    # declaring variable
    # to implement
    # exponential search
    pos = N + 1

    # list to store
    # last N lines
    lines = []

    # opening file using with() method
    # so that file get closed
    # after completing work
    with open(fname) as f:

        # loop which runs
        # until size of list
        # becomes equal to N
        while len(lines) <= N:

            # try block
            try:
                # moving cursor from
                # left side to
                # pos line from end
                f.seek(-pos, 2)

            # exception block
            # to handle any run
            # time error
            except IOError:
                f.seek(0)
                break

            # finally block
            # to add lines
            # to list after
            # each iteration
            finally:
                lines = list(f)

            # increasing value
            # of variable
            # exponentially
            pos *= 2

    # returning the
    # whole list
    # which stores last
    # N lines
    return lines[-N:]

def get_echelle_params(path_to_file: str) -> dict:
    """
    Read the spectrometer settings from the footer of the file
    Parameters
    ----------
    path_to_file: str
        The path to the echelle file

    Returns
    -------
    dict:
        A dictionary containing the spectrometer settings used in the experiment
    """

    footer = LastNlines(path_to_file, 40)
    params = {}
    for line in footer:
        matches = f_pattern.match(line)
        if not matches is None:
            params[matches.group(1)] = matches.group(2)
    return params

def load_echelle_file(path_to_file: str) -> Tuple[pd.DataFrame, dict]:
    """
    Reads the data form the echelle file and the parameters from the footer

    Parameters
    ----------
    path_to_file: str
        The path to the echelle file

    Returns
    -------
    pd.DataFrame, tuple
        A pandas DataFrame containing the spectrum, and a dictionary with the spectrometer settings.
    """
    df: pd.DataFrame = pd.read_csv(
        path_to_file,  sep=r'\s+', engine='python',
        usecols=[0, 1],
        names=['wl (nm)', 'counts']
    ).apply(pd.to_numeric, errors='coerce').dropna()
    # read the last 40 lines of the file
    footer = LastNlines(path_to_file, 40)

    params = {}
    for line in footer:
        matches = f_pattern.match(line)
        if not matches is None:
            params[matches.group(1)] = matches.group(2)
    return df, params