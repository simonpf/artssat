"""
artssat.sensor.utils
--------------------

Helper functions for handling satellite sensors.
"""
import numpy as np
import scipy as sp

def sensor_properties(center_frequencies, offsets, order = "positive"):
    """
    Compute ARTS frequency grid array and sensor response matrix for
    for paired side-band channels from a list of given center frequencies
    and corresponding offsets.

    Args:

        center_frequencies: List of center frequencies of the sensor

        offsets: List of lists of offsets for each sensor frequency

        order: "negative" if the sum of center frequencies and negative
            offset should be used to order the frequencies. Positive
            otherwise.
    """
    i = 0 # output index, increased only for <order> offsets.
    j = 0 # f_grid index, increased for each offset

    f_grid = []
    sr_data = []
    sr_i    = []
    sr_j    = []
    ci = 0

    for f, ofs in zip(center_frequencies, offsets):
        ofs.sort()

        nfs = len(ofs)

        for i, o in enumerate(ofs[::-1]):
            if o > 0.0:
                f_grid  += [f - o]
                sr_data += [0.5]
                sr_j    += [j]
                if order == "negative":
                    sr_i    += [ci + i]
                else:
                    sr_i += [ci + nfs - 1 - i]
                j += 1

        if ofs[0] == 0.0:
            f_grid  += [f]
            sr_data += [1.0]
            sr_j    += [j]
            if order == "negative":
                sr_i += [ci + i]
            else:
                sr_i += [ci]
            j += 1

        for i, o in enumerate(ofs):
            if o > 0.0:
                f_grid  += [f + o]
                sr_data += [0.5]
                sr_j    += [j]
                if order == "negative":
                    sr_i    += [ci + nfs - 1 - i]
                else:
                    sr_i += [ci + i]
                j += 1

        ci += nfs
    sensor_response = sp.sparse.coo_matrix((sr_data, (sr_i, sr_j)))

    return np.array(f_grid), sensor_response
