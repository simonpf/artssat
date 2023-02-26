"""
Tests the Field07 PSD.
"""
import numpy as np
from artssat.scattering.psd.f07 import F07


def test_f07():
    """
    Ensure that manual calculate of moments yields similar result
    to parametrization used by the Field07 PSD.
    """
    psd = F07(1e-3, 1.0, 2.0)
    t = -30
    x = np.logspace(-6, -2, 201)
    psd_data = psd.evaluate(x, t)

    m2 = psd_data.get_moment(2)

    assert np.isclose(m2, 1e-3, rtol=1e-2)
