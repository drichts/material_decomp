import os
import numpy as np
from interp_attenuation import log_interp_1d


def filter_spectrum(spectrum, material, thickness):
    """
    Uses Beer's law to filter a spectra through a metal filter of specified thickness
    :param material: str
            The chemical symbol of the filter material, e.g. 'Cu' or 'Al'
    :param thickness: int, float
            The thickness of the filter in mm
    :return: spectrum, the filtered spectrum
    """
    directory = r'Material_decomposition_data\K-edge Decomposition\Filter Attenuation'

    # Material attenuation (with Raylegh scattering)
    mat_atten = np.loadtxt(os.path.join(directory, f'{material}_linear.txt'))

    # Interpolate for the energies in the spectrum
    interp_func = log_interp_1d(mat_atten[:, 0], mat_atten[:, 1])
    interp_mat_atten = interp_func(spectrum[:, 0])

    # Convert thickness to cm (and add in the negative for Beer's Law)
    thickness = -1 * thickness / 10

    # Filter
    spectrum[:, 1] = spectrum[:, 1] * np.exp(interp_mat_atten * thickness)

    return spectrum