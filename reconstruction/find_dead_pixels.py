import numpy as np


def find_dead_pixels(data1, data2, dark):
    """
    This function will take your attenuation data from a PCD acquisition and find the dead pixels and return the mask
    of the dead pixels
    :param data1: ndarray. An airscan, same shape as data2, should be acquired over the same length of time.
            Shape: <24, 576>
    :param data2: ndarray. An airscan, same shape as data1, should be acquired over the same length of time.
            Shape: <24, 576>
    :param dark: ndarray. A dark scan, same shape as data1 and data2. Should be acquired under no x-ray flux.
            Shape: <24, 576>
    :return: mask
            2D Mask with np.nan where there are dead pixels, 1's at every other pixel location
    """

    # Compute the log ratio between the two airscans
    dpm = np.abs(np.log(data1) - np.log(data2)) * 100

    data = np.add(data1, data2)

    # Find the median over the entire length of the airscan (data1 plus data2)
    med_data = np.nanmedian(data, axis=(0, 1))
    min_data = med_data / 2
    max_data = med_data * 2

    # Any data outside of the range [min_data, max_data] set to np.nan
    dpm[data > max_data] = np.nan
    dpm[data < min_data] = np.nan

    # Any residual pixels over 0.5 set to np.nan, everything else set to 1's
    dpm[dpm >= 0.5] = np.nan
    dpm[dpm < 0.5] = 1

    # Find any pixels in the dark scan that have excessive counts
    dark[dark >= 50] = np.nan
    dark[dark < 50] = 1

    # Combine the dead pixel masks
    dpm = dpm * dark

    return dpm

