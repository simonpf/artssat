"""
The MY05 particle size distribution as proposed by Milbrandt and Yau in [my05]_.

.. [my05] Milbrandt, J. A., and M. K. Yau. "A multimoment bulk microphysics
    parameterization. Part I: Analysis of the role of the spectral
    shape parameter." Journal of the atmospheric sciences
    62.9 (2005): 3051-3064.
"""
import numpy as np

class MY05:
    def __init__(mass_density, number_density, alpha):
        """
        Create instance of the MY05 PSD from given mass density, number
        density and alpha value

        """

        self.mass_density   = np.reshape(mass_density, (-1, 1))

        shape = self.mass_density.shape

        try:
            self.number_density = np.broadcast_to(number_density, shape)
        except:
            raise Exception("Could not broadcast number density data to the"
                            " shape of the mass density data.")

        try:
            self.alpha = np.broadcast_to(alpha, shape)
        except:
            raise Exception("Could not broadcast alpha data to shape"
                            "of the mass density data.")

