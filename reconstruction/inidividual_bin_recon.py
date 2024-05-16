import os
import numpy as np
import tigre.algorithms as algs
from reconstruction.pcd_geom import PCDGeometry
import general_functions as gen


def correct_dead_pixels(data, dpm, one_bin, one_frame):
    """
    Correct the miscalibrated pixels in the projection data
    :param data: ndarray, the projection data; shape: <angles, rows, cols>
    :param dpm: ndarray, the array showing the locations of bad pixels with np.nan in their location. 1's are good
            pixels. Shape: <rows, cols>
    :param one_bin: boolean; If the data only has 1 energy bin, set to to True, else False
    :param one_frame: boolean: if the data only has 1 angle or projection set to True, elseh False
    :return: data, the projection data with bad pixels corrected
    """

    data = gen.correct_dead_pixels(data, one_frame=one_frame, one_bin=one_bin, dead_pixel_mask=dpm)

    data = gen.correct_leftover_pixels(data, one_frame=one_frame, one_bin=one_bin)

    return data


def reconstruct_CT(sinogram, filt='hamming', h_offset=0):
    """
    Reconstruct a CT image from the sinogram
    :param sinogram: ndarray, the projection data of the CT scan; shape: <angles, rows, cols>
    :param filt: string, Fourier filter for the sinogram. Default: 'hamming'
    :param h_offset: int, float; the horizontal detector offset. Default: 0
    :return: ct, the reconstructed CT image
    """

    # Create the reconstruction geometry
    geo = PCDGeometry(h_offset=h_offset)

    # Set the angles for reconstruction
    num_proj = len(sinogram)
    angles = np.linspace(0, 2 * np.pi, num_proj, endpoint=False)

    # Convert the sinogram to float32 if necessary
    sinogram = np.float32(sinogram)

    # Reconstruct
    print(f'Starting reconstruction using FDK')
    ct = algs.fdk(sinogram, geo, angles, filter=filt)

    return ct


def normalize_ct(ct, water_mask=None, water_slice=15, water_norm_val=None):
    """
    Normalize a CT image to HU
    :param ct: ndarray, the CT data, shape: <slices, rows, cols>
    :param water_mask: ndarray, the mask showing where the water is in the image
            shape: <rows, cols>; contains 1's in the pixels with the object, np.nan elsewhere
    :param water_slice: int, the slice in which to find the raw values of water
    :param water_norm_val: int, float; if using a separate scan to find the raw value of water, input that value here
            Default: None
    :return: ct, the noramlized CT image (to HU)
    """

    slices, rows, cols = np.shape(ct)

    if not water_norm_val:
        # Find the water values for each of the bins
        water_value = np.nanmean(ct[water_slice] * water_mask)
    else:
        water_value = water_norm_val

    # Normalize the data
    ct = 1000 / water_value * np.subtract(ct, water_value)

    xx, yy, = np.mgrid[:cols, :rows]
    circle = (xx - rows // 2) ** 2 + (yy - cols // 2) ** 2

    # Set any area outside the field of view to -1000 HU (air)
    radius = 240 / 512 * rows

    for z in range(slices):
        ct[z][circle > radius ** 2] = -999

    return ct


def reconstruct_kedge(sino_low, sino_high, energies, material='Au', filter='hamming', h_offset=0):
    """
    Reconstruct a K-edge CT image
    :param sino_low: ndarray, the low energy sinogram data; shape <angles, rows, cols>
    :param sino_high: ndarray, the high energy sinogram data; shape <angles, rows, cols>
    :param energies: ndarray, three values in order from lowest to highest of the three energy thresholds defining
            the two energy bins. e.g. {65, 81, 97} (in keV)
    :param material: string, the K-edge material (use the elemental abbreviation), default: 'Au'
    :param filter: string, the filter applied to the sinogram data in the Fourier space. Default: 'hamming'
    :param h_offset: int, float; the horizontal offset of the detector
    :return: sinogram, ct: the K-edge sinogram and K-edge CT iamges
    """
    # Fetch the material and water mass attenuation coefficients
    directory = r'Material_decomposition_data\K-edge Decomposition'
    mat_att = np.loadtxt(os.path.join(directory, 'K-edge materials', f'{material}.txt'))
    water_att = np.loadtxt(os.path.join(directory, 'Background materials', 'H2O.txt'))

    # Translate from MeV to keV for the energies
    mat_att[:, 0] = mat_att[:, 0] * 1000
    water_att[:, 0] = water_att[:, 0] * 1000

    # Look for the closest energy to the 3 thresholds
    idx = []

    for energy in energies:
        idx.append(np.argmin(np.abs(water_att[:, 0] - energy)))

    low_mat = np.mean(mat_att[idx[0]:idx[1] + 1, 1])
    high_mat = np.mean(mat_att[idx[1]:idx[2] + 1, 1])
    low_water = np.mean(water_att[idx[0]:idx[1] + 1, 1])
    high_water = np.mean(water_att[idx[1]:idx[2] + 1, 1])

    # Convert the sinogram to float32 if necessary
    sino_high = np.float32(sino_high) * low_water
    sino_low = np.float32(sino_low) * high_water

    # K-edge decomposition
    sinogram = (sino_high - sino_low) / ((high_mat * low_water) - (low_mat * high_water))

    sinogram = np.squeeze(gen.correct_leftover_pixels(sinogram, one_frame=False, one_bin=True))

    # Create the reconstruction geometry
    geo = PCDGeometry(h_offset=h_offset)

    # Set the angles for reconstruction
    num_proj = len(sinogram)
    angles = np.linspace(0, 2 * np.pi, num_proj, endpoint=False)

    # Reconstruct
    print(f'Starting reconstruction using FDK')
    ct = algs.fdk(sinogram, geo, angles, filter=filter)

    return sinogram, ct


def normalize_kedge(ct, norm_slice=8, high_conc_real=50, concentration_vals=None, contrast_mask=None, water_mask=None):
    """
    Normalize a reconstructed K-edge CT image between 0 mg/ml (water) and a high concentration
    :param ct: ndarray, the CT data, shape: <slices, rows, cols>
    :param norm_slice: int, the slice from which to take the mean values to calibrate to
    :param high_conc_real: int, float; the high concentration (in mg/ml) to calibrate to
    :param concentration_vals: 2d array; if using a separate image for the raw mean values of the low and high
            concentration data, this will have two floats of shape [high raw concentration value, low raw value]
    :param contrast_mask: ndarray, the mask showing where the object with high concentration is in the image
            shape: <rows, cols>; contains 1's in the pixels with the object, np.nan elsewhere
    :param water_mask: ndarray, the mask showing where the water is in the image
            shape: <rows, cols>; contains 1's in the pixels with the object, np.nan elsewhere
    :return: ct, the normalized image
    """
    # Normalize the K-edge data to the concentration values (in another image) that are given in conc_vals
    # or the image itself

    # These values are the image values of the highest and 0% concentrations vials before normalization
    if concentration_vals is not None:
        low_conc_val_img = concentration_vals[1]  # The mean value in of water in the image
        high_conc_val_img = concentration_vals[0]  # The mean value of the high concentration material
    else:
        low_conc_val_img = np.nanmean(ct[norm_slice] * water_mask)
        high_conc_val_img = np.nanmean(ct[norm_slice] * contrast_mask)

    # Normalize
    ct = (ct - low_conc_val_img) / (high_conc_val_img - low_conc_val_img) * high_conc_real

    ct[ct < 0] = 0

    return ct

