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
                                   \right )^\beta \right \}

The parameter X is defined as  the volume equivalent sphere diameter
:math:`D_{eq}` normalized by the mass-weighted mean diameter:

.. math::

    X = \frac{D_{eq}}{D_m}

The PSD is thus parametrized by four parameters:
    - :math:`N_0^*`, here called the *intercept parameter*
    - :math:`D_m`, the *mass-weighted mean diameter*
    - the shape parameters :math:`\alpha` and :math:`\beta`

Of these, :math:`\alpha` and :math:`\beta` are generally assumed fixed, while
:math:`N_0` and :math:`D_m` are the predictive parameters that describe
the distribution of particles withing a given atmospheric volume.

The particle mass density :math:`m` per bulk volume can be computed
from :math:`N_0` and :math:`D_m` using:

.. math::
    m = \frac{\Gamma(4)}{4^4}\frac{\pi \rho}{6}N_0^*D_m^4

In this module, two implementations of the D14 PSD are provided:

- the :class:`D14` class that uses the mass-density and :math:`D_m` as
  moments of the PSD
- the :class:`D14N` :class that uses the intercept parameter :math:`N_0^*`
  and :math:`D_m` as moments of the PSD

"""
from artssat import dimensions as dim
from artssat.scattering.psd.arts.arts_psd import ArtsPSD
from artssat.scattering.psd.data.psd_data import PSDData, D_eq
from pyarts.workspace import arts_agenda
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

    n0 = np.reshape(n0, result_shape)

    try:
        dm = np.broadcast_to(dm, shape).reshape(result_shape)
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
    x = x / dm

    c1 = gamma(4.0) / 4 ** 4
    c2 = gamma((alpha + 5) / beta) ** (4 + alpha) / \
         gamma((alpha + 4) / beta) ** (5 + alpha)
    c3 = gamma((alpha + 5) / beta) / \
         gamma((alpha + 4) / beta)

    y = n0 * beta * c1 * c2
    y = y * x ** alpha
    y *= np.exp(- (x * c3) ** beta)

    # Set invalid values to zero
    y[np.broadcast_to(dm == 0.0, y.shape)] = 0.0

    return y

################################################################################
# PSD classes
################################################################################

class D14(ArtsPSD):
    """

    Implementation of the D14 PSD that uses mass density :math:`m` and
    mass-weighted mean diameter :math:`D_m` as free parameters.
    """

    @classmethod
    def from_psd_data(self, psd, alpha, beta, rho):
        """
        Create an instance of the D14 PSD from existing PSD data.

        Parameters:

            :code:`psd`: A numeric or analytic representation of
                a PSD.

            alpha(:class:`numpy.ndarray`): The :math:`\alpha` parameter of
            the to use for the D14 PSD.

            beta(:class:`numpy.ndarray`): The :math:`\beta` parameter of
            the to use for the D14 PSD.

            rho(:class:`numpy.float`): The average density of the hydrometeors,
            should be somewhere in between :math:`916.7 kg\m^{-3}` and
            :math:`1000 kg\m^{-3}`.
        """
        new_psd = D14(alpha, beta, rho)
        new_psd.convert_from(psd)
        return new_psd

    def convert_from(self, psd):
        """
        Converts a given psd to a :class:`D14` PSD with the :math:`\alpha, \beta`
        and :math:`\rho` this :class`D14` instance.

        Arguments:

            psd: Another psd object providing :code:`get_mass_density` and
            `get_moment` member functions to compute moments of the PSD.
        """
        md = psd.get_mass_density()

        m4 = psd.get_moment(4.0, reference_size_parameter = self.size_parameter)
        m3 = psd.get_moment(3.0, reference_size_parameter = self.size_parameter)

        dm = m4 / m3
        dm[m3 == 0.0] = 0.0

        self.mass_density = md
        self.mass_weighted_diameter = dm

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
        from artssat.scattering.psd.data.psd_data import D_eq

        self.alpha = alpha
        self.beta  = beta
        self.rho = rho

        if not mass_density is None:
            self.mass_density = mass_density
        if not mass_weighted_diameter is None:
            self.mass_weighted_diameter = mass_weighted_diameter

        super().__init__(D_eq(self.rho))

        self.rho = rho
        self.dm_min = 1e-12

    @property
    def moment_names(self):
        return ["mass_density", "mass_weighted_diameter"]

    @property
    def moments(self):
        return [self.mass_density, self.mass_weighted_diameter]

    @property
    def pnd_call_agenda(self):
        @arts_agenda
        def pnd_call(ws):
            ws.psdDelanoeEtAl14(n0Star = -999.0,
                                Dm     = np.nan,
                                iwc    = np.nan,
                                rho    = self.rho,
                                alpha  = self.alpha,
                                beta   = self.beta,
                                t_min  = self.t_min,
                                dm_min = self.dm_min,
                                t_max  = self.t_max)
        return pnd_call

    def _get_parameters(self):

        md = self.mass_density
        if md is None:
            raise Exception("The 'mass_density' array needs to be set to use"
                            "this function.")

        shape = md.shape

        dm = self.mass_weighted_diameter
        if dm is None:
            raise Exception("The 'mass_weighted_diameter' array needs to be set "
                            "to use this function.")

        try:
            dm = np.broadcast_to(dm, shape)
        except:
            raise Exception("Could not broadcast the 'mass_weighted_diameter'"
                            "data into the shape of the mass density data.")

        try:
            alpha = np.broadcast_to(self.alpha, shape)
        except:
            raise Exception("Could not broadcast the data for the 'alpha' "
                            " parameter  into the shape the mass density data.")

        try:
            beta = np.broadcast_to(self.beta, shape)
        except:
            raise Exception("Could not broadcast the data for the 'beta' "
                            " parameter  into the shape the mass density data.")

        return md, dm, alpha, beta

    def get_moment(self, p, reference_size_parameter = None):
        """
        Computes the moments of the PSD analytically.

        Parameters:

            p(:code:`numpy.float`): Wich moment of the PSD to compute

            reference_size_parameter(:class:`SizeParameter`): Size parameter
            with respect to which the moment should be computed.

        Returns:

            Array containing the :math:`p` th moment of the PSD.

        """
        if not reference_size_parameter is None:
            a1 = self.size_parameter.a
            b1 = self.size_parameter.b
            a2 = reference_size_parameter.a
            b2 = reference_size_parameter.b

            c = (a1 / a2) ** (p / b2)
            p = p * b1 / b2
        else:
            c = 1.0

        md, dm, alpha, beta = self._get_parameters()
        n0 = 4.0 ** 4 / (np.pi * self.rho) * md / dm ** 4.0

        nu_mgd    = beta
        lmbd_mgd  = gamma((alpha + 5) / beta) / \
                    gamma((alpha + 4) / beta)
        alpha_mgd = (alpha + 1) / beta - 1
        n_mgd = n0 * gamma(4.0) / 4.0 ** 4 * \
                gamma((alpha + 1) / beta) * \
                gamma((alpha + 5) / beta) ** 3 / \
                gamma((alpha + 4) / beta) ** 4

        m = n_mgd / lmbd_mgd ** p
        m *= gamma(1 + alpha_mgd + p / nu_mgd)
        m /= gamma(1 + alpha_mgd)

        return c * m * dm ** (p + 1)

    def get_mass_density(self):
        """
        Returns:
            Array containing the mass density for all the bulk volumes described
            by this PSD.
        """
        if self.mass_density is None:
            raise Exception("The free mass_density parameter has not been set.")
        else:
            return self.mass_density

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

        try:
            md = self.mass_density
        except:
            raise Exception("The 'mass_density' array needs to be set,  before"
                            " the PSD can be evaluated.")

        try:
            dm = self.mass_weighted_diameter
        except:
            raise Exception("The 'mass_weighted_diameter' array needs to be"
                            " set,  before the PSD can be evaluated.")

        n0 = 4.0 ** 4 / (np.pi * self.rho) * md / dm ** 4.0

        y =  evaluate_d14(x, n0, dm, self.alpha, self.beta)
        return PSDData(x, y, D_eq(self.rho))

class D14N(ArtsPSD):
    """

    Implementation of the D14 PSD that uses the intercept parameter :math:`N_0^*`
    and the mass-weighted mean diameter :math:`D_m` as free parameters.

    """

    @classmethod
    def from_psd_data(cls, psd, alpha, beta, rho):
        """
        Create an instance of the D14 PSD from existing PSD data.

        Parameters:
            :code:`psd`: A numeric or analytic representation of
                a PSD.

            alpha(:code:`numpy.ndarray`): The :math:`alpha` parameter of
            the to use for the D14 PSD.

            beta(:code:`numpy.ndarray`): The :math:`beta` parameter of
            the to use for the D14 PSD.

            rho(:code:`numpy.float`): The density to use for the D14 PSD
        """
        new_psd = cls(alpha, beta, rho)
        new_psd.convert_from(psd)
        return new_psd

    def convert_from(self, psd):

        md = psd.get_mass_density()

        m4 = psd.get_moment(4.0, reference_size_parameter = self.size_parameter)
        m3 = psd.get_moment(3.0, reference_size_parameter = self.size_parameter)

        dm = m4 / m3
        dm[m3 == 0.0] = 0.0
        n0 = 4.0 ** 4 / (np.pi * self.rho) * md / dm ** 4
        n0[m3 == 0.0] = 0.0

        self.mass_density = md
        self.intercept_parameter = n0
        self.mass_weighted_diameter = dm


    def __init__(self, alpha, beta, rho = 917.0,
                 intercept_parameter = None,
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
        from artssat.scattering.psd.data.psd_data import D_eq

        self.alpha = alpha
        self.beta  = beta
        self.rho = rho

        if not intercept_parameter is None:
            self.intercept_parameter = intercept_parameter
        if not mass_weighted_diameter is None:
            self.mass_weighted_diameter = mass_weighted_diameter

        self.dm_min = 1e-12

        super().__init__(D_eq(self.rho))

    @property
    def moment_names(self):
        return ["intercept_parameter", "mass_weighted_diameter"]

    @property
    def moments(self):
        try:
            return [self.intercept_parameter, self.mass_weighted_diameter]
        except:
            return None

    @property
    def pnd_call_agenda(self):
        @arts_agenda
        def pnd_call(ws):
            ws.psdDelanoeEtAl14(n0Star = np.nan,
                                Dm     = np.nan,
                                iwc    = -999.0,
                                rho    = self.rho,
                                alpha  = self.alpha,
                                beta   = self.beta,
                                t_min  = self.t_min,
                                dm_min = self.dm_min,
                                t_max  = self.t_max)
        return pnd_call

    def _get_parameters(self):

        n0 = self.intercept_parameter
        if n0 is None:
            raise Exception("The 'intercept_parameter' data needs to be set to "
                            " use this function.")

        shape = n0.shape

        dm = self.mass_weighted_diameter
        if dm is None:
            raise Exception("The 'mass_weighted_diameter' array needs to be set "
                            "to use this function.")

        try:
            dm = np.broadcast_to(dm, shape)
        except:
            raise Exception("Could not broadcast the 'mass_weighted_diameter'"
                            "data into the shape of the mass density data.")

        try:
            alpha = np.broadcast_to(self.alpha, shape)
        except:
            raise Exception("Could not broadcast the data for the 'alpha' "
                            " parameter  into the shape the mass density data.")

        try:
            beta = np.broadcast_to(self.beta, shape)
        except:
            raise Exception("Could not broadcast the data for the 'beta' "
                            " parameter  into the shape the mass density data.")

        return n0, dm, alpha, beta


    def get_mass_density(self):
        """
        Returns:
            Array containing the mass density for all the bulk volumes described
            by this PSD.
        """
        if self.intercept_parameter is None \
           or self.mass_weighted_diameter is None :
            raise Exception("The parameters of the PSD have not been set.")
        else:
            c = gamma(4.0) / 4.0 ** 4.0
            m = c * np.pi * self.rho / 6.0 * self.intercept_parameter \
                * self.mass_weighted_diameter ** 4.0
            return m

    def get_moment(self, p, reference_size_parameter = None):
        """
        Computes the moments of the PSD analytically.

        The physical significance of a moment of a PSD depends on the size
        parameter. So in general, the moments of the same PSD given w.r.t.
        different size parameters differ. If the
        :code:`reference_size_parameter` argument is given then the
        computed moment will correspond to the Moment of the PSD w.r.t. to
        the given size parameter.

        Parameters:

            p(:code:`numpy.float`): Wich moment of the PSD to compute

            reference_size_parameter(SizeParameter): Size parameter with
            respect to which the moment should be computed.

        Returns:

            Array containing the :math:`p` th moment of the PSD.

        """
        if not reference_size_parameter is None:
            a1 = self.size_parameter.a
            b1 = self.size_parameter.b
            a2 = reference_size_parameter.a
            b2 = reference_size_parameter.b

            c = (a1 / a2) ** (p / b2)
            p = p * b1 / b2
        else:
            c = 1.0

        n0, dm, alpha, beta = self._get_parameters()

        nu_mgd    = beta
        lmbd_mgd  = gamma((alpha + 5) / beta) / \
                    gamma((alpha + 4) / beta)
        alpha_mgd = (alpha + 1) / beta - 1
        n_mgd = n0 * gamma(4.0) / 4.0 ** 4 * \
                gamma((alpha + 1) / beta) * \
                gamma((alpha + 5) / beta) ** 3 / \
                gamma((alpha + 4) / beta) ** 4

        m = n_mgd / lmbd_mgd ** p
        m *= gamma(1 + alpha_mgd + p / nu_mgd)
        m /= gamma(1 + alpha_mgd)

        return c * m * dm ** (p + 1)

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
        n0 = self.intercept_parameter
        if n0 is None:
            raise Exception("The 'intercept_parameter' array needs to be set,  before"
                            " the PSD can be evaluated.")

        dm = self.mass_weighted_diameter
        if dm is None:
            raise Exception("The 'mass_weighted_diameter' array needs to be"
                            " set,  before the PSD can be evaluated.")

        y =  evaluate_d14(x, n0, dm, self.alpha, self.beta)
        return PSDData(x, y, D_eq(self.rho))

class D14MN(D14N):
    """

    Implementation of the D14 PSD that uses mass density $m$ and intercept
    parameter :math:`N_0^*` as free parameters.

    """
    def __init__(self, alpha, beta, rho = 917.0,
                 mass_density = None,
                 intercept_parameter = None):
        """
        Parameters:

            alpha(numpy.float): The value of the :math:`alpha` parameter for
                the PSD

            beta(numpy.float): The value of the :math:`beta` parameter for
                the PSD

            rho(numpy.float): The particle density to use for the conversion
                to mass density.

            mass_density(numpy.array): If provided, this can be used to fix
                the mass density which will then not be queried from the data
                provider.

            intercept_parameter(numpy.array): If provided, this can be used to fix
                the value of the intercept parameter $N_0^*$ which will then not
                be queried from the data provider.

        """
        from artssat.scattering.psd.data.psd_data import D_eq

        if (not mass_density is None) and (not intercept_parameter is None):
            self.mass_density = mass_density
            dm = (4.0 ** 4 / np.pi / rho * mass_density / intercept_parameter) ** (1 / 4.0)
        else:
            dm = None

        super().__init__(alpha, beta, rho, intercept_parameter, dm)

    @property
    def moment_names(self):
        return ["mass_density", "intercept_parameter"]

    @property
    def moments(self):
        return [self.mass_density, self.intercept_parameter]

    @property
    def pnd_call_agenda(self):
        @arts_agenda
        def pnd_call(ws):
            ws.psdDelanoeEtAl14(n0Star = np.nan,
                                Dm     = -999.0,
                                iwc    = np.nan,
                                rho    = self.rho,
                                alpha  = self.alpha,
                                beta   = self.beta,
                                t_min  = self.t_min,
                                dm_min = self.dm_min,
                                t_max  = self.t_max)
        return pnd_call

    def _get_parameters(self):

        md = self.mass_density
        if md is None:
            raise Exception("The 'intercept_parameter' data needs to be set to "
                            " use this function.")
        shape = md.shape

        n0 = self.intercept_parameter
        if n0 is None:
            raise Exception("The 'intercept_parameter' data needs to be set to "
                            " use this function.")

        dm = (4.0 ** 4 / np.pi / self.rho * md / n0) ** 0.25


        try:
            alpha = np.broadcast_to(self.alpha, shape)
        except:
            raise Exception("Could not broadcast the data for the 'alpha' "
                            " parameter  into the shape the mass density data.")

        try:
            beta = np.broadcast_to(self.beta, shape)
        except:
            raise Exception("Could not broadcast the data for the 'beta' "
                            " parameter  into the shape the mass density data.")

        return n0, dm, alpha, beta


    def get_mass_density(self):
        """
        Returns:
            Array containing the mass density for all the bulk volumes described
            by this PSD.
        """
        return self.mass_density

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
        n0, dm, alpha, beta = self._get_parameters()
        y =  evaluate_d14(x, n0, dm, alpha, beta)
        return PSDData(x, y, D_eq(self.rho))
