import numpy as np


def find_nearest_index(data, value):
    """
    This function will return the index of the array element that is closest to the value
    :param data: ndarray
            The data to search for the element closest to the value
    :param value: int, float
            The value to search for in the array
    :return: int
            The index of the element that is closest to the value
    """

    return np.argmin(np.abs(data - value))
