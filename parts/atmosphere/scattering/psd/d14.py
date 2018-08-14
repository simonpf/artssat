"""
The D14 particle size distribution as proposed by Delanoë in [d14]_.

.. [d14] Delanoë, J. M. E., et al. "Normalized particle size distribution
    for remote sensing application." Journal of Geophysical Research:
    Atmospheres 119.7 (2014): 4204-4227.
"""

class D14:
    properties = [("mass_density", (dim.p, dim.lat, dim.lon), np.ndarray),
                  ("intercept_parameter", (dim.p, dim.lat, dim.lon), np.ndarray),
                  ("mass_weighted_diameter", (dim.p, dim.lat, dim.lon), np.ndarray),
                  ("alpha", (), np.float),
                  ("beta", (), np.float),
                  ("rho", (), np.float),
                  ("t_min", (), 0.0),
                  ("t_max", (), 999.0)]

    def __init__(self, alpha, beta,
                 mass_density = None,
                 intercept_parameter = None,
                 mass_weighted_diameter = None,
                 rho = 917.0):

        self.alpha = alpha
        self.beta  = beta

        if (not mass_density is None) and (not intercept_parameter is None) \
           and (not mass_weighted_diameter is None):
            raise Exception("This is a 2-moment PSD, so only two parameters of"
                            " the parameters 'mass_density', "
                            "'intercept_parameter' and 'mass_weighted_diameter'"
                            " can be set.")

        if not mass_density is None:
            self.mass_density = mass_density

        if not intercept_parameter is None:
            self.intercept_parameter is None

        if not mass_weighted_diameter is None:
            self.mass_weighted_diameter = mass_weighted diameter

        self.rho = rho
        self.t_min = 0.0
        self.t_max = 999.0

