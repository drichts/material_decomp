import numpy as np


def correct_dead_pixels(data, one_frame, one_bin, dead_pixel_mask):
    """
    This is to correct for known dead pixels. Takes the average of the eight surrounding pixels.
    Could implement a more sophisticated algorithm here if needed.
    :param data: 4D ndarray
                The data array in which to correct the pixels <captures, rows, columns, counter>,
                <captures, rows, columns> if there is only 1 set of data, or
                <rows, columns> for just one frame of one set of data
    :param one_frame: boolean
                True if there is only one bin, False if not
    :param one_bin: boolean
                True if there is only one bin, False if not
    :param dead_pixel_mask: 2D ndarray
                A data array with the same number of rows and columns as 'data'. Contains np.nan everywhere there
                is a known non-responsive pixel
    :return: The data array corrected for the dead pixels
    """

    if one_frame:
        data = np.expand_dims(data, axis=0)
    if one_bin:
        data = np.expand_dims(data, axis=3)
        dead_pixel_mask = np.expand_dims(dead_pixel_mask, axis=2)

    # Find the dead pixels (i.e pixels = to nan in the DEAD_PIXEL_MASK)
    dead_pixels = np.array(np.argwhere(np.isnan(dead_pixel_mask)), dtype='int')

    for p_idx, pixel in enumerate(dead_pixels):

        # Pixel is corrected in every counter and capture
        data[:, pixel[0], pixel[1], pixel[2]] = get_average_pixel_value(data, pixel, dead_pixel_mask)

    return np.squeeze(data)


def get_average_pixel_value(img, pixel, dead_pixel_mask):
    """
    Averages the dead pixel using the 8 nearest neighbours
    Checks the dead pixel mask to make sure each of the neighbors is not another dead pixel

    :param img: 4D array
                The projection image. Shape: <frames, rows, columns, bins>

    :param pixel: tuple (row, column)
                The problem pixel (is a 2-tuple)

    :param dead_pixel_mask: 2D numpy array
                Mask with 1 at good pixel coordinates and np.nan at bad pixel coordinates

    :return: the average value of the surrounding pixels
    """
    shape = np.shape(img)
    row, col, b = pixel

    vals = np.zeros((8, shape[0]))

    # Count the number of nans around the pixel, if above a certain number we'll include the 16 pixels surrounding the
    # immediate 8 pixels surrounding the pixel, it will be less if the pixel is on an edge
    num_nans = 0
    edge = False  # If the pixel is on edge or not (True if it is)

    # Grabs each of the neighboring pixel values and sets to nan if they are bad pixels or
    # outside the bounds of the image
    if col == shape[2] - 1:
        vals[0] = np.nan
        num_nans += 1
    else:
        vals[0] = img[:, row, col + 1, b] * dead_pixel_mask[row, col + 1, b]
        if np.isnan(dead_pixel_mask[row, col + 1, b]):
            num_nans += 1
    if col == 0:
        vals[1] = np.nan
        num_nans += 1
    else:
        vals[1] = img[:, row, col - 1, b] * dead_pixel_mask[row, col - 1, b]
        if np.isnan(dead_pixel_mask[row, col - 1, b]):
            num_nans += 1
    if row == shape[1] - 1:
        vals[2] = np.nan
        num_nans += 1
    else:
        vals[2] = img[:, row + 1, col, b] * dead_pixel_mask[row + 1, col, b]
        if np.isnan(dead_pixel_mask[row + 1, col, b]):
            num_nans += 1
    if row == 0:
        vals[3] = np.nan
        num_nans += 1
    else:
        vals[3] = img[:, row - 1, col, b] * dead_pixel_mask[row - 1, col, b]
        if np.isnan(dead_pixel_mask[row - 1, col, b]):
            num_nans += 1
    if col == shape[2] - 1 or row == shape[1] - 1:
        vals[4] = np.nan
        num_nans += 1
    else:
        vals[4] = img[:, row + 1, col + 1, b] * dead_pixel_mask[row + 1, col + 1, b]
        if np.isnan(dead_pixel_mask[row + 1, col + 1, b]):
            num_nans += 1
    if col == 0 or row == shape[1] - 1:
        vals[5] = np.nan
        num_nans += 1
    else:
        vals[5] = img[:, row + 1, col - 1, b] * dead_pixel_mask[row + 1, col - 1, b]
        if np.isnan(dead_pixel_mask[row + 1, col - 1, b]):
            num_nans += 1
    if col == shape[2] - 1 or row == 0:
        vals[6] = np.nan
        num_nans += 1
    else:
        vals[6] = img[:, row - 1, col + 1, b] * dead_pixel_mask[row - 1, col + 1, b]
        if np.isnan(dead_pixel_mask[row - 1, col + 1, b]):
            num_nans += 1
    if col == 0 or row == 0:
        vals[7] = np.nan
        num_nans += 1
    else:
        vals[7] = img[:, row - 1, col - 1, b] * dead_pixel_mask[row - 1, col - 1, b]
        if np.isnan(dead_pixel_mask[row - 1, col - 1, b]):
            num_nans += 1

    # Takes the average of the neighboring pixels excluding nan values
    avg = np.nanmean(vals, axis=0)

    return avg


def correct_leftover_pixels(data, one_frame, one_bin):
    """
    This will correct for any other nan or inf pixels
    :param data: 4D ndarray
                The data array in which to correct the pixels <captures, rows, columns, counter>,
                <captures, rows, columns> if there is only 1 set of data, or
                <rows, columns> for just one frame of one set of data
    :param one_frame: boolean
                True if there is only one bin, False if not
    :param one_bin: boolean
                True if there is only one bin, False if not
    :param dpm: 2D ndarray
                A data array with the same number of rows and columns as 'data'. Contains np.nan everywhere there
                is a known non-responsive pixel
    :return: The data array corrected for the dead pixels
    """

    if one_frame:
        data = np.expand_dims(data, axis=0)
    if one_bin:
        data = np.expand_dims(data, axis=3)

    # This will find any left over nan values and correct them
    data[np.isinf(data)] = np.nan  # Set inf values to nan
    nan_coords = np.argwhere(np.isnan(data))
    num_nan = len(nan_coords)
    while num_nan > 0:
        print(f'Correcting secondary nan coords: {num_nan} left')
        for c_idx, coords in enumerate(nan_coords):
            coords = tuple(coords)
            frame = coords[0]
            pixel = coords[-3:-1]
            img_bin = coords[-1]
            temp_img = data[frame, :, :, img_bin]

            data[coords] = get_average_single_pixel_value(temp_img, pixel, np.ones((24, 576)))

        nan_coords = np.argwhere(np.isnan(data))
        print(f'After correction: {len(nan_coords)} dead pixels left')
        print()
        # If we can't correct for the remaining coordinates break the loop
        if len(nan_coords) == num_nan:
            print(f'Broke because the number of nan pixels remained the same: {num_nan}')
            break
        num_nan = len(nan_coords)

    return data
