import numpy as np
from scipy import interpolate


def log_interp_1d(xx, yy, kind="linear"):
    """
    Perform interpolation in log-log scale.
    Args:
        xx (List[float]): x-coordinates of the points.
        yy (List[float]): y-coordinates of the points.
        kind (str or int, optional):
            The kind of interpolation in the log-log domain. This is passed to
            scipy.interpolate.interp1d.
    Returns:
        A function whose call method uses interpolation
        in log-log scale to find the value at a given point.
    """
    log_x = np.log(xx)
    log_y = np.log(yy)
    # No big difference in efficiency was found when replacing interp1d by
    # UnivariateSpline
    lin_interp = interpolate.interp1d(log_x, log_y, kind=kind)
    return lambda zz: np.exp(lin_interp(np.log(zz)))
