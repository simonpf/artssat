r"""
The Delanoë (2014) PSD
======================

The D14 particle size distribution as proposed by Delanoë in :cite:`delanoe2014`
uses a normalized form of the modified gamma distribution, parametrized
as follows:

.. math::

    \frac{dN(X)}{dX} = N_0^* \beta \frac{\Gamma(4)}{4^4}
           \frac{\Gamma(\frac{\alpha + 5}{\beta})^{(4 + \alpha)}}
                {\Gamma(\frac{\alpha + 4}{\beta})^{(5 + \alpha)}}
    X^\alpha \exp \left \{- \left (X \frac{\Gamma(\frac{\alpha + 5}{\beta})}
                                        {\Gamma(\frac{\alpha + 4}{\beta})}
                                   \right )^\beta \right \},

The parameter X is defined as  the volume equivalent sphere diameter
:math:`D_{eq}` normalized by the mass-weighted mean diameter:

.. math::

    X = \frac{D_{eq}}{D_m}

The PSD is thus parametrized by four parameters:
    - :math:`N_0^*`, here called the *intercept parameter*
    - :math:`D_m`, the *mass-weighted mean diameter*
    - the shape parameters :math:`\alpha` and :math:`\beta`

Of these, :math:`\alpha` and :math:`\beta` are assumed constant, while
:math:`N_0` and :math:`D_m` are the free parameters that descibe
the particles throughout the atmosphere.

The particle mass density :math:`\rho_m` per bulk volume can be computed
from :math:`N_0` and :math:`D_m` using:

.. math::
    \rho_m = \frac{\Gamma(4)}{4^4}\frac{\pi \rho}{6}N_0^*D_m^4

In this module, two implementations of the D14 PSD are provided:

- the :class:`D14` class that uses the mass-density and :math:`D_m` as
  moments of the PSD
- the :class:`D14N` :class that uses the intercept parameter :math:`N_0^*`
  and :math:`D_m` as moments of the PSD

"""
from parts import dimensions as dim
from parts.arts_object import ArtsObject
from parts.scattering.psd.arts.arts_psd import ArtsPSD
from parts.scattering.psd.data.psd_data import PSDData
import numpy as np
import scipy as sp
from scipy.special import gamma

################################################################################
# General PSD function
################################################################################

def evaluate_d14(x, n0, dm, alpha, beta):
    """
    Compute the particle size distribution of the D14 PSD.



    Parameters:

        x(numpy.array): 1D array containing the values of the size parameter
            :math:`D_{eq}` at which to evaluate the PSD. If :code:`x` is not
            1D it will be flattened.

        n0(numpy.array or scalar): Array containing the values of the
            intercept parameter for which the PSD should be evaluated.

        dm(numpy.array or scalar): Array containing the values of the mass
            weighted mean diameter at which to evaluate the PSD. Must be
            broadcastable to the shape of :code:`n0`

        alpha(numpy.array or scalar): Array containing the values of the
            :math:`alpha` parameter a which to evaulate the PSD. Must be
            broadcastable to the shape of :code: `n0`

        beta(numpy.array or scalar): Array containing the values of the
            :math:`beta` parameter a which to evaulate the PSD. Must be
            broadcastable to the shape of :code: `n0`

    Returns:

        Array :code:`dNdD_eq` containing the computed values of the PSD. The first
        dimensions of :code:`dNdD_eq` correspond to the shape of the :code:`n0`
        parameter and the last dimension to the size parameter.

    """
    shape = n0.shape
    result_shape = shape + (1,)

    n0 = np.reshape(result_shape)

    try:
        np.broadcast_to(dm, shape).reshape(result_shape)
    except:
        raise Exception("Could not broadcast 'dm' parameter to shape of 'n0' "
                        "parameter.")

    try:
        alpha = np.broadcast_to(alpha, shape).reshape(result_shape)
    except:
        raise Exception("Could not broadcast 'alpha' parameter to shape of 'n0' "
                        "parameter.")

    try:
        beta = np.broadcast_to(beta, shape).reshape(result_shape)
    except:
        raise Exception("Could not broadcast 'beta' parameter to shape of 'n0' "
                        "parameter.")

    x = x.reshape((1,) * len(shape) + (-1,))

    c1 = gamma(4) / 4 ** 4
    c2 = gamma((alpha + 5) / beta) ** (4 + alpha) / \
         gamma((alpha + 4) / beta) ** (5 + alpha)

    y = n0 * beta * c1 * c2
    y *= x ** alpha
    y *= np.exp(- (x * c2) ** beta)

    return y

################################################################################
# PSD classes
################################################################################

class D14(ArtsPSD, metaclass = ArtsObject):
    """

    Implementation of the D14 PSD that uses mass density :math:`\rho_m`  and
    the mass-weighted mean diamter :math:`D_m` as free parameters.

    """
    properties = [("mass_density", (dim.p, dim.lat, dim.lon), np.ndarray),
                  ("mass_weighted_diameter", (dim.p, dim.lat, dim.lon), np.ndarray),
                  ("alpha", (), np.float),
                  ("beta", (), np.float),
                  ("rho", (), np.float)]

    def __init__(self, alpha, beta, rho = 917.0,
                 mass_density = None,
                 mass_weighted_diameter = None):
        """
        Parameters:

            alpha(numpy.float): The value of the :math:`alpha` parameter for
                the PSD

            beta(numpy.float): The value of the :math:`beta` parameter for
                the PSD

            rho(numpy.float): The particle density to use for the conversion
                to mass density.

            mass_density(numpy.array): If provided, this can be used to fix
                the value of the mass density which will then not be queried
                from the data provider.

            mass_weighted_diameter(numpy.array): If provided, this can be used
                to fix the value of the mass weighted mean diameter which will
                then not be queried from the data provider.

        """
        from parts.scattering.psd.data.psd_data import SizeParameter

        self.alpha = alpha
        self.beta  = beta
        self.rho = rho

        if mass_density:
            self.mass_density = mass_density
        if mass_weighted_diameter:
            self.mass_weighted_diameter = mass_weighted_diameter

        super().__init__(alpha, beta, SizeParameter.D_eq)

        self.rho = rho
        self.t_min = 0.0
        self.t_max = 999.0

    @property
    def moment_names(self):
        return ["mass_density", "mass_weighted_diameter"]

    @property
    def pnd_call_agenda(self):
        @arts_agenda
        def pnd_call(ws):
            ws.psdD14(n0Star = -999.0,
                      Dm     = np.nan,
                      iwc    = np.nan,
                      rho    = self.rho,
                      alpha  = self.alpha,
                      beta   = self.beta,
                      t_min  = self.t_min,
                      Dm_min = D14.dm_min,
                      t_max  = self.t_max)

    def evaluate(self, x):
        """
        Compute value of the particle size distribution for given values of the
        size parameter.

        Parameters:
            x(numpy.array): Array containing the values of :math:`D_eq` at which to
                compute the number density.

        Returns:

            Array :code:`dNdD_eq` containing the computed values of the PSD. The first
            dimensions of :code:`dNdD_eq` correspond to the shape of the :code:`n0`
            parameter and the last dimension to the size parameter.

        """
        md = self.mass_density
        if md is None:
            raise Exception("The 'mass_density' array needs to be set,  before"
                            " the PSD can be evaluated.")

        dm = self.mass_weighted_diameter
        if dm is None:
            raise Exception("The 'mass_weighted_diameter' array needs to be"
                            " set,  before the PSD can be evaluated.")

        n0 = 4.0 ** 4 / (np.pi * self.rho) * md / dm ** 4.0

        return PSDData(evaluate_d14(x, n0, dm, alpha, beta),
                       x,
                       SizeParameter.D_eq)


class D14N(ArtsPSD, metaclass = ArtsObject):
    """

    Implementation of the D14 PSD that uses the intercept parameter :math:`N_0^*`
    and the mass-weighted mean diamter :math:`D_m` as free parameters.

    """
    properties = [("intercept_parameter", (dim.p, dim.lat, dim.lon), np.ndarray),
                  ("mass_weighted_diameter", (dim.p, dim.lat, dim.lon), np.ndarray),
                  ("alpha", (), np.float),
                  ("beta", (), np.float),
                  ("rho", (), np.float)]

    def __init__(self, alpha, beta, rho = 917.0,
                 intercept_paramter = None,
                 mass_weighted_diameter = None):
        """
        Parameters:

            alpha(numpy.float): The value of the :math:`alpha` parameter for
                the PSD

            beta(numpy.float): The value of the :math:`beta` parameter for
                the PSD

            rho(numpy.float): The particle density to use for the conversion
                to mass density.

            intercept_parameter(numpy.array): If provided, this can be used to fix
                the value of the mass density which will then not be queried
                from the data provider.

            mass_weighted_diameter(numpy.array): If provided, this can be used
                to fix the value of the mass weighted mean diameter which will
                then not be queried from the data provider.

        """
        from parts.scattering.psd.data.psd_data import SizeParameter

        self.alpha = alpha
        self.beta  = beta
        self.rho = rho

        if intercept_parameter:
            self.intercept_parameter = intercept_parameter
        if mass_weighted_diameter:
            self.mass_weighted_diameter = mass_weighted_diameter

        super().__init__(alpha, beta, SizeParameter.D_eq)

        self.rho = rho
        self.t_min = 0.0
        self.t_max = 999.0

    @property
    def moment_names(self):
        return ["intercept_parameter", "mass_weighted_diameter"]

    @property
    def pnd_call_agenda(self):
        @arts_agenda
        def pnd_call(ws):
            ws.psdD14(n0Star = np.nan,
                      Dm     = np.nan,
                      iwc    = -999.0,
                      rho    = self.rho,
                      alpha  = self.alpha,
                      beta   = self.beta,
                      t_min  = self.t_min,
                      Dm_min = D14.dm_min,
                      t_max  = self.t_max)

    def evaluate(self, x):
        """
        Compute value of the particle size distribution for given values of the
        size parameter.

        Parameters:
            x(numpy.array): Array containing the values of :math:`D_eq` at which to
                compute the number density.

        Returns:

            Array :code:`dNdD_eq` containing the computed values of the PSD. The first
            dimensions of :code:`dNdD_eq` correspond to the shape of the :code:`n0`
            parameter and the last dimension to the size parameter.

        """
        md = self.intercept_parameter
        if n0 is None:
            raise Exception("The 'intercept_parameter' array needs to be set,  before"
                            " the PSD can be evaluated.")

        dm = self.mass_weighted_diameter
        if dm is None:
            raise Exception("The 'mass_weighted_diameter' array needs to be"
                            " set,  before the PSD can be evaluated.")

        n0 = 4.0 ** 4 / (np.pi * self.rho) * md / dm ** 4.0

        return PSDData(evaluate_d14(x, n0, dm, alpha, beta),
                       x,
                       SizeParameter.D_eq)
