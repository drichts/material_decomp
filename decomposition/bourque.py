import numpy as np
from numpy.linalg import LinAlgError


def solve_for_c(z_calib, hu_low, hu_high, K=3, L_method='DEI'):
    """
    Calculate the c coefficients based on the known Z value (z_calib), the two HU values, and the order of the
    polynomial (K)
    :param z_calib: float, the known Z value of the material
    :param hu_low: float, the mean HU value of the material in the low energy scan
    :param hu_high: float, the mean HU value of the material in the high energy scan
    :param K: int, the order of the polynomial, default: 3
    :param method: string, the method to calculate the Dual energy index (default: DEI)
    :return: c, vector of calibration coefficients
    """

    # Adjust the HU values to reduced HU
    u_low = hu_low / 1000 + 1
    u_high = hu_high / 1000 + 1

    # Calculate L based either on DEI or DER
    if L_method == 'DEI':
        L = np.divide(u_low - u_high, u_low + u_high)
    else:
        L = np.divide(u_low, u_high)

    # Construct the full L matrix
    L = np.transpose(np.tile(L, (K, 1)))
    L = np.power(L, np.arange(K))

    # Solve for c
    try:
        c, c_res, c_rank, c_s = np.linalg.lstsq(L, z_calib)
        return c
    except LinAlgError:
        print('c did not converge')
        return None

def solve_for_z(hu_low, hu_high, c, L_method='DEI', air_HU=-900):
    """
    Calculate the Z value based on given parameters.
    :param hu_low: Lower bound of Hounsfield Unit (HU) range or array of values.
    :param hu_high: Upper bound of Hounsfield Unit (HU) range or array of values.
    :param c: Coefficients.
    :param L_method: Method for calculating the L matrix. Default is 'DEI'.
    :return: Calculated Z value or array of values.
    """
    # Find the K order of c
    k = len(c)

    # Adjust the HU values to reduced HU
    u_low = hu_low / 1000 + 1
    u_high = hu_high / 1000 + 1

    # Calculate the L matrix based on the values of hu_low and hu_high
    if L_method == 'DEI':
        L = np.divide(u_low - u_high, u_low + u_high)
    else:
        L = np.divide(u_low, u_high)

    # Create the L matrix for this specific material
    L_matrix = np.power(np.tile(L, (k, 1)).T, np.arange(k))

    # # Calculate Z for each instance
    Z = np.sum(np.multiply(c, L_matrix), axis=1)

    Z[(hu_low < air_HU) | (hu_high < air_HU)] = 7.66

    return Z


def solve_for_b(z_calib, rho_calib, hu, M=3):
    """
    Calculate the b calibration coefficients based on the known Z and electron density values along with the mean
    HU value of the material
    :param z_calib: float, the known Z value of the material
    :param rho_calib: float, the known electron density value of the material
    :param hu: float, the mean HU value of the material in the scan
    :param M: int, the order of the polynomial, default: 3
    :return: b, vector of calibration coefficients
    """

    # Adjust the HU values to reduced HU
    u = hu / 1000 + 1

    # Create the F matrix, first the Z matrix (with powers), rho is the rho vector in each column
    Z = np.transpose(np.tile(z_calib, (M, 1)))
    Z = np.power(Z, np.arange(M))
    rho = np.transpose(np.tile(rho_calib, (M, 1)))
    F = np.multiply(rho, Z)

    # Solve for b_low and b_high
    try:
        b, b_res, b_rank, b_s = np.linalg.lstsq(F, u)
        return b
    except LinAlgError:
        print('b did not converge')
        return None

def solve_for_rho(z_val, hu_low, hu_high, b_low, b_high, air_HU=-900):
    """
    Calculate rho based on given parameters.

    :param z_val: Z values or array of Z values.
    :param hu_low: Lower bound of Hounsfield Unit (HU) range or array of values.
    :param hu_high: Upper bound of Hounsfield Unit (HU) range or array of values.
    :param b_low: Coefficients for low energy or array of coefficients.
    :param b_high: Coefficients for high energy or array of coefficients.
    :return: Calculated rho or array of rho values.
    """

    # Adjust the HU values to reduced HU
    u_low = hu_low / 1000 + 1
    u_high = hu_high / 1000 + 1

    # Calculate the value of M
    M = len(b_low)

    # Construct the Z matrix with appropriate powers
    Z_matrix = np.power(np.tile(z_val, (M, 1)).T, np.arange(M))

    # Calculate rho_low and rho_high for each instance
    rho_low = u_low / np.sum(np.multiply(b_low, Z_matrix), axis=1)
    rho_high = u_high / np.sum(np.multiply(b_high, Z_matrix), axis=1)

    # Calculate the appropriate rho for each instance
    rho = 0.5 * (rho_low + rho_high)

    rho[(hu_low < air_HU) | (hu_high < air_HU)] = 0.0011

    return rho
