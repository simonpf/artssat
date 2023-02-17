import numpy as np


def e_eq_water_mk(temp):
    r"""
    Equilibrium vapor pressure from Murphy and Koop (2005).

    Copied from the typhon package.

    Args:
        temperature: The temperature in Kelvin

    Returns:
        The equlibrium water vapor in Kelvin.
    """
    if np.any(temp <= 0):
        raise Exception("Temperatures must be larger than 0 Kelvin.")

    e = (
        54.842763
        - 6763.22 / temp
        - 4.21 * np.log(temp)
        + 0.000367 * temp
        + np.tanh(0.0415 * (temp - 218.8))
        * (53.878 - 1331.22 / temp - 9.44523 * np.log(temp) + 0.014025 * temp)
    )

    return np.exp(e)


def relative_humidity2vmr(rh, press, temp):
    r"""Convert relative humidity into water vapor VMR.

    Args:
        rh: The relative humidity.
        press: The pressure in Pa
        temp: The temperature in K

    Returns:
        The corresponding VMR of water vapro.
    """

    return rh * e_eq_water_mk(temp) / press


def vmr2relative_humidity(vmr, press, temp):
    r"""Convert water vapor VMR into relative humidity.

    Parameters:
        vmr: Volume mixing ratio of water vapor.
        press: The pressure in Pa
        temp: The temperature in K.

    Returns:
        The corresponding relative humidity.
    """

    return vmr * press / e_eq_water_mk(temp)
