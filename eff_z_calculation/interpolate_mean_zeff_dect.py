# Calculate the effective atomic number for the Gammex materials for DECT

import os
import numpy as np
import pandas as pd
from filter_spectrum import filter_spectrum
from interp_attenuation import log_interp_1d
from scipy.interpolate import interp1d
from find_nearest import find_nearest_index


def order_array_by_indices(array, indices):
    ordered_array = [array[i] for i in indices]
    return ordered_array


def interp_mean_zeff(low_t1, low_t2, high_t1, high_t2, material):
    """
    Interpolate the effective atomic number based on the energy thresholds or beam spectra
    :param low_t1: int, lower threshold of the low energy bin or spectrum
    :param low_t2: int, higher threshold of the low energy bin or spectrum
    :param high_t1: int, lower threshold of the high energy bin or spectrum
    :param high_t2: int, higher threshold of the high energy bin or spectrum
    :param material: string, material type for the Gammex phantom
    :return: the mean and standard deviation of the effective Z values
    """

    spec_folder = r'Material_decomposition_data\K-edge Decomposition\Beam Spectrum'
    folder = r'Material_decomposition_data\K-edge Decomposition\Material Decomposition Inserts\Elements'
    elements = ['H', 'O', 'C', 'N', 'Cl', 'Ca', 'P', 'Mg', 'Si']
    z = [1, 8, 6, 7, 17, 20, 15, 12, 14]

    directory = r'Material_decomposition_data\K-edge Decomposition\Material Decomposition Inserts'
    mat_folder = os.path.join(directory, 'Ideal')

    mat_info = pd.read_csv(os.path.join(mat_folder, 'Elemental Weights.csv'))

    beam120 = np.load(os.path.join(spec_folder, 'corrected-spectrum_120kV.npy'))
    beam80 = np.load(os.path.join(spec_folder, '80kV.npy'))

    beam120 = filter_spectrum(beam120, 'Al', 6)
    energies120 = beam120[:, 0]
    spectrum120 = beam120[:, 1]

    energies80 = beam80[:, 0] / 1000 # Convert to MeV
    spectrum80 = beam80[:, 1]

    z_values = np.zeros(2)
    z_idx = 0

    for t1, t2 in [[low_t1, low_t2], [high_t1, high_t2]]:

        if t2 == 80:
            energies = energies80
            t1_idx = find_nearest_index(energies, t1 / 1000)  # Low bin first threshold energy index
            t2_idx = find_nearest_index(energies, t2 / 1000)  # Low bin second threshold energy index

            temp_spectrum = np.copy(spectrum80)

            temp_spectrum[:t1_idx] = 0
            temp_spectrum[t2_idx + 1:] = 0
        else:
            energies = energies120
            t1_idx = find_nearest_index(energies, t1 / 1000)  # Low bin first threshold energy index
            t2_idx = find_nearest_index(energies, t2 / 1000)  # Low bin second threshold energy index

            temp_spectrum = np.copy(spectrum120)

            temp_spectrum[:t1_idx] = 0
            temp_spectrum[t2_idx + 1:] = 0

        cross_section_theoretical = np.zeros(len(elements))
        for eidx, elem in enumerate(elements):

            cs = np.loadtxt(os.path.join(folder, f'elec_{elem}.txt'))

            interp_func = log_interp_1d(cs[:, 0], cs[:, 1])
            cs = interp_func(energies)

            # Average of the elemental cross section over the spectrum
            cross_section_theoretical[eidx] = np.average(cs, weights=temp_spectrum)

        sort_order = np.argsort(z)
        z_sorted = np.sort(z)
        cross_section_theoretical = order_array_by_indices(cross_section_theoretical, sort_order)

        csf = interp1d(cross_section_theoretical, z_sorted, kind='cubic')

        # The individual elemental weights
        elem_weights = mat_info[material].values[2:] / 100

        cross_section = 0

        for eidx, elem in enumerate(elements):

            cs = np.loadtxt(os.path.join(folder, f'elec_{elem}.txt'))

            # Interpolate the for energies in the spectrum
            interp_func = log_interp_1d(cs[:, 0], cs[:, 1])
            cs = interp_func(energies)

            cs = cs * elem_weights[eidx]

            cross_section += np.average(cs, weights=temp_spectrum)

        z_values[z_idx] = csf(cross_section)
        z_idx += 1

    return np.array([np.mean(z_values), np.abs(z_values[0] - z_values[1])/2])


if __name__ == '__main__':

    materials = ['Water', 'LN-300 Lung', 'LN-450 Lung', 'BR-12 Breast', 'AP6 Adipose', 'Solid Water', 'LV1 Liver',
                 'BRN-SR2 Brain', 'IB Inner Bone', 'B-200 Bone', 'CB2-30%', 'CB2-50%', 'SB3 Cortical Bone']

    for mat in materials:
        me, de = interp_mean_zeff(24, 80, 24, 120, mat)
        print(mat, f'{me:0.3f}', f'{de:0.3f}')
    print()
