r"""
The Milbrandt-Yau (2005) PSD
============================

The Milbrandt-Yau (2005) PSD used in the GEM model uses a modified
gamma distribution of the form:

.. math::
    N(D_{max}) = N_0 D_{max}^\nu \exp(- \lambda D_{max} ^ \mu)

In this two-moment scheme, the number density :math:`\rho_n` and mass
density :math:`\rho_m`  are used as predictive moments. They can be related
to the :math:`N_0` and :math:`\lambda` parameters of the PSD using

.. math::
    \rho_n = \frac{N_0}{\mu} \lambda^{\frac{\nu}{\mu}}
             \Gamma (\frac{\nu + 1}{\mu})

    \rho_m = a\frac{N_0}{\mu} \lambda^{\frac{\nu + b}{\mu}}
             \Gamma ( \frac{\nu + 1 + b}{\mu} ),

where :math:`a, b` are the parameters of the mass-size relationship

.. math::
    M(D) = a \cdot D_{max}^b.

"""
import numpy as np
import scipy as sp
from scipy.special import gamma
from pyarts.workspace import arts_agenda

from artssat import dimensions as dim
from artssat.scattering.psd.modified_gamma     import ModifiedGamma
from artssat.scattering.psd.data.psd_data      import D_max
from artssat.scattering.psd.arts.arts_psd import ArtsPSD
from artssat.scattering.psd.data.psd_data import PSDData



settings = {"cloud_ice" : (0.0, 1.0, 440., 3.0),
            "rain" : (0.0, 1.0, 523.5988, 3.0),
            "snow" : (0.0, 1.0, 52.35988, 3.0),
            "graupel" : (0.0, 1.0, 209.4395, 3.0),
            "hail" : (0.0, 1.0, 471.2389, 3.0),
            "cloud_water" : (1.0, 1.0, 523.5988, 3.0)}

class MY05(ArtsPSD):
    r"""
    The :class:`MY05` class describes the size distributions of particles
    in an atmosphere using the number density :math:`\rho_n` and mass
    density :math:`\rho_m` as predictive moments.

    The :math:`\nu` and :math:`\mu` parameters take on fixed values depending
    on the hydrometeor type.
    """

    properties = [("number_density", (dim.p, dim.lat, dim.lon), np.ndarray),
                  ("mass_density", (dim.p, dim.lat, dim.lon), np.ndarray),
                  ("nu", (), np.float),
                  ("mu", (), np.float)]

    @classmethod
    def from_psd_data(self, psd, nu, mu, a, b):
        r"""
        Create a MY05 PSD from given psd data.

        Parameters:

            psd(PSDData or other PSD): PSD data from which to create the MY05
            representation.

            nu(:code:`float`): :math:`\nu` parameter of MY05 PSD

            mu(:code:`float`): :math:`\mu` parameter of MY05 PSD

            a(:code:`float`): :math:`a` coefficient of the mass-size relationship

            b(:code:`float`): :math:`b` coefficient of the mass-size relationship

            hydrometeor_type: One of ["cloud_ice", "rain", "snow", "graupel", "hail"]
                or None. If not None this will override the given settings for
                nu, mu, a, b.
        """
        size_parameter = D_max(a, b)
        number_density = psd.get_moment(0)
        mass_density   = psd.get_mass_density()

        return MY05(nu, mu, a, b, None, number_density, mass_density)

    def __init__(self,
                 nu = None,
                 mu = None,
                 a  = None,
                 b  = None,
                 hydrometeor_type = None,
                 number_density = None,
                 mass_density = None):
        r"""
        Parameters:
            nu(:code:`numpy.float`): The :math:`\nu` parameter of the PSD

            mu(:code:`numpy.float`): The :math:`\mu` parameter of the PSD

            a(:code:`numpy.float`): The :math:`a` coefficient of the
            mass-size realtionship.

            b(:code:`numpy.float`): The :math:`b` coefficient of the
            mass-size realtionship.

            number_density(:code:`numpy.ndarray`): Array containing
            the number density for a given set of volume elements in an
            atmosphere.

            mass_density(:code:`numpy.ndarray`): Array containing
            the mass density for a given set of volume elements in an
            atmosphere.
        """
        if not hydrometeor_type is None:
            if hydrometeor_type in settings:
                nu, mu, a, b = settings[hydrometeor_type]
            else:
                raise Exception("Expected keyword hydrometeor type to be one of"
                                " {0} but got '{1}'.".format(list(settings.keys()),
                                                           hydrometeor_type))

        self.nu = nu
        self.mu = mu

        self.hydrometeor_type = hydrometeor_type

        if not number_density is None:
            self.number_density = number_density

        if not mass_density is None:
            self.mass_density = mass_density

        super().__init__(D_max(a, b))

    def convert_from(self, psd):
        r"""
        Convert given psd to MY05 PSD with :math:`\nu,\mu,a` and :math:`b`
        parameters of this instance.

        Parameters:

            psd: Other PSD providing :code:`get_moment` and :code:`get_mass_density`
            member functions.
        """
        self.number_density = psd.get_moment(0)
        self.mass_density   = psd.get_mass_density()

    def _get_parameters(self):
        """
        Checks if parameters of the PSD are available and tries to broadcast
        them to the shape of the number density data. Converts the mass density
        and number density data to the :math:`N_0` and :math:`\lambda`
        parameters of the PSD function.

        Returns:

            :code:`tuple(n0, lmbd, mu, nu)` containing the four parameters of
            the PSD function.

        Raises:

            An exception if any of the parameters is not set or cannot be
            broadcasted into the shape of the number density data.
        """

        # Number density
        n = self.number_density
        if n is None:
           raise Exception("The number density needs to be set to use"
                            " this function.")
        shape = n.shape

        # Mass density
        m = self.mass_density
        if m is None:
           raise Exception("The mass density needs to be set to use"
                            " this function.")
        try:
            m = np.broadcast_to(m, shape)
        except:
            raise Exception("Could not broadcast mass density to the shape"
                            "of the provided intercept parameter N.")

        # Alpha parameter
        try:
            mu = np.broadcast_to(self.mu, shape)
        except:
            raise Exception("Could not broadcast alpha paramter to the shape"
                            "of the provided intercept parameter N.")

        # Nu parameter
        try:
            nu = np.broadcast_to(self.nu, shape)
        except:
            raise Exception("Could not broadcast nu paramter to the shape"
                            "of the provided intercept parameter N.")

        a = self.size_parameter.a
        b = self.size_parameter.b

        lmbd = (a * n) / m \
               * gamma((nu + 1 + b) / mu) \
               / gamma((nu + 1) / mu)
        lmbd = lmbd ** (mu / b)

        inds = np.logical_or(n == 0.0, m == 0.0)
        lmbd[inds] = 0.0

        n0 = n * mu * lmbd ** ((nu + 1.0) / mu) / gamma((nu + 1.0) / mu)
        n0[inds] = 0.0

        return n0, lmbd, mu, nu

    @property
    def moment_names(self):
        """
        The names of the predictive moments of the PSD.
        """
        return ["number_density", "mass_density"]

    @property
    def moments(self):
        try:
            return [self.number_density, self.mass_density]
        except:
            return None

    @property
    def pnd_call_agenda(self):
        """
        The ARTS WSM implementing the MY05 PSD.
        """
        @arts_agenda
        def pnd_call(ws):
            ws.psdMilbrandtYau05(hydrometeor_type = self.hydrometeor_type,
                                 t_min = self.t_min,
                                 t_max = self.t_max)
        return pnd_call

    def get_moment(self, p, reference_size_parameter = None):
        r"""
        Analytically computes the :math:`p` th moment :math:`M(p)` of the PSD
        using

        .. math::

            M(p) = \frac{N_0}{\mu} \lambda ^{-\frac{\nu + p + 1}{\mu}}
                   \Gamma ( \frac{\nu + p + 1}{\mu})

        Parameters:

            p(:code:`float`): Which moment of the distribution to compute.

        Returns:

            :code:`numpy.ndarray` containing the :math:`p` th moment for
            all volume elements described by the PSD.

            reference_size_parameter(:class: `SizeParameter`): Size parameter
            with respect to which the moment should be computed.

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

        n0, lmbd, mu, nu = self._get_parameters()
        m = n0 / mu * lmbd ** (-(nu + p + 1) / mu) * gamma((nu + 1.0 + p) / mu)

        m[lmbd == 0.0] = 0.0

        return c * m

    def get_mass_density(self):
        r"""
        Returns:

             The :code:`numpy.ndarray` containing the mass density data of
             the PSD.

        """
        a = self.size_parameter.a
        b = self.size_parameter.b
        return a * self.get_moment(b)

    def evaluate(self, x):
        r"""
        Compute a numeric representation of the PSD data.

        Parameters:

            x(:code:`numpy.ndarray`): Array containing the values of the size
            parameter at which to evaluate the PSD.

        Returns:

            :code:`PSDData` object containing the numeric PSD data corresponding
            to this PSD.

        """
        n0, lmbd, mu, nu = self._get_parameters()

        shape = n0.shape
        result_shape = shape + (1,)

        n0   = np.reshape(n0, result_shape)
        lmbd = np.broadcast_to(lmbd, shape).reshape(result_shape)
        mu   = np.broadcast_to(mu, shape).reshape(result_shape)
        nu   = np.broadcast_to(nu, shape).reshape(result_shape)

        x = x.reshape((1,) * len(shape) + (-1,))

        y = n0 * x ** nu * np.exp(- lmbd * x ** mu)
        return PSDData(x, y, self.size_parameter)
