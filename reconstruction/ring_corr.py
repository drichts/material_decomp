import numpy as np
from scipy.signal import medfilt
from general_functions import correct_dead_pixels, correct_leftover_pixels


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end part of the output signal.
    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
    output:
        the smoothed signal
    example:
    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    see also:
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(), s, mode='valid')
    return y


def devon_correction(data, air, dpm, num_bins=7):
    """
    Creates a gain correction map for each of the pixels in the detector based on a water phantom scan
    :param data: The water phantom scan; shape <rows, cols, bins>
    :param air: The corresponding air or flat field scan; shape <rows, cols, bins>
    :param dpm: The dead pixel correction mask; shape <rows, cols, bins>
    :param num_bins: The number of bins, default: 7
    :return:
    """
    # Correct the water projections for air
    print('Correcting dead pixels in water scan')
    if num_bins == 1:
        one_bin = True
    else:
        one_bin = False
    data = np.squeeze(correct_dead_pixels(data, one_frame=True, one_bin=one_bin, dead_pixel_mask=dpm))

    corr = np.squeeze(np.log(air + 0.01) - np.log(data + 0.01))

    corr = correct_leftover_pixels(np.squeeze(corr), one_frame=True, one_bin=one_bin)
    corr = np.squeeze(corr)

    if one_bin:
        corr = np.expand_dims(corr, axis=2)

    num_rows = np.shape(corr)[0]
    num_cols = np.shape(corr)[1]

    # Create a smoothed signal
    corrected_array = np.zeros((num_rows, num_cols, num_bins))

    xpts = np.arange(num_cols)
    for row in range(num_rows):
        for bb in range(num_bins):

            med_row = medfilt(corr[row, :, bb], 21)  # Median filter of the current row

            p = np.polyfit(xpts, med_row, 8)

            corrected_array[row, :, bb] = np.polyval(p, xpts)

    # The correction array for all other data is the ratio of the corrected over the old data
    corr_array = np.divide(corrected_array, corr)

    return np.squeeze(corr_array)

