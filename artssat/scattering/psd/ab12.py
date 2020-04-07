r"""
The Abel-Boutle (2012) PSD
==========================

The Abel-Boutle (2012) PSD is a single moment PSD intended to represent rain
drops. Particle number densities are represented using a gamma distribution function

.. math::

    N(D) &= N_0\ D^\gamma \ \exp(-\lambda D).

The parameters :math:`N_0` and :math:`\lambda` can be diagnosed from the rain water
content using

.. math::

    \lambda &= \left [ \frac{\pi \rho_\text{w} x_1 \Gamma(4 + \mu)}{6 \rho_\text{air} q_\text{R}}]^{\frac{1}{4 + \mu - x_2}}
    N_0 &= x_1 \lambda^{x_2}


.. [AB2012] Abel SJ, Boutle IA. 2012. An improved representation of the raindrop size distribution for
single-moment microphysics schemes. Q. J. R. Meteorol. Soc. 138: 2151–2162. DOI:10.1002/qj.1949

"""
import numpy as np
import scipy as sp
from scipy.special import gamma
from pyarts.workspace import arts_agenda

from artssat import dimensions as dim
from artssat.scattering.psd.data.psd_data import D_eq
from artssat.scattering.psd.arts.arts_psd import ArtsPSD
from artssat.scattering.psd.data.psd_data import PSDData



class AB12(ArtsPSD):
    r"""
    The AB12 class provides an implementation of the Abel-Boutle (2012) single-moment
    PSD for rain drops.
    """

    @classmethod
    def from_psd_data(self, psd, mu = 0.0):
        r"""
        Create a AB12 PSD from given psd data.

        Parameters:

            psd(PSDData or other PSD): PSD data from which to create the MY05
            representation.
            mu(:code:`float` or array): The value of the mu parameter to use.
        """
        mass_density   = psd.get_mass_density()
        return AB12(mu, mass_density)

    def __init__(self,
                 mu = 0.0,
                 mass_density = None):
        r"""
        Parameters:
            mu(:code:`numpy.float`): The :math:`\mu` parameter of the PSD
            mass_density(:code:`numpy.ndarray`): Array containing
            the water content for a given set of volume elements in an
            atmosphere.
        """
        self.mu = mu

        if not mass_density is None:
            self.mass_density = mass_density

        super().__init__(D_eq(1000.0))

    def convert_from(self, psd):
        r"""
        Convert given psd to AB12 PSD with :math:`\mu` parameter of this instance.

        Parameters:

            psd: Other PSD providing :code:`get_moment` and :code:`get_mass_density`
            member functions.
        """
        self.mass_density   = psd.get_mass_density()

    def _get_parameters(self):
        """
        Checks if parameters of the PSD are available and tries to broadcast
        them to the shape of the mass density data. Calculates parameters of

        Returns:

            :code:`tuple(n0, lmbd, mu)` containing the  parameters of
            the PSD function.

        Raises:

            An exception if any of the parameters is not set or cannot be
            broadcasted into the shape of the number density data.
        """

        # Number density

        # Mass density
        m = self.mass_density
        if m is None:
           raise Exception("The mass density needs to be set to use"
                            " this function.")
        shape = m.shape

        try:
            mu = np.broadcast_to(np.array(self.mu), shape)
        except:
            raise Exception("Could not broadcast mu paramter to the shape"
                            "of the provided intercept parameter N.")


        x1 = 0.22
        x2 = 2.2
        lmbd = (np.pi * 1000.0 * x1 * gamma(4 + mu)) / (6.0 *  m)
        lmbd = lmbd ** (1.0 / (4.0 + mu - x2))
        print(lmbd)
        n0 = x1 * lmbd ** x2

        return n0, lmbd, mu

    @property
    def moment_names(self):
        """
        The names of the predictive moments of the PSD.
        """
        return ["mass_density"]

    @property
    def moments(self):
        try:
            return [self.mass_density]
        except:
            return None

    @property
    def pnd_call_agenda(self):
        """
        The ARTS SM implementing the MY05 PSD.
        """
        @arts_agenda
        def pnd_call(ws):
            ws.psdAbelBoutle12(t_min = self.t_min,
                               t_max = self.t_max)
        return pnd_call

    def get_moment(self, p, reference_size_parameter = None):
        r"""
        Analytically computes the :math:`p` th moment :math:`M(p)` of the PSD
        using

        .. math::

            M(p) = N_0  \lambda^{-(p + \mu + 1)}
                   \Gamma (p + 1 + \mu)

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

        n0, lmbd, mu = self._get_parameters()
        m = n0 * lmbd ** (-(mu + p + 1)) * gamma(mu + 1.0 + p)

        m[lmbd == 0.0] = 0.0

        return c * m

    def get_mass_density(self):
        r"""
        Returns:

             The :code:`numpy.ndarray` containing the mass density data of
             the PSD.

        """
        return self.mass_density

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
        n0, lmbd, mu = self._get_parameters()

        shape = n0.shape
        result_shape = shape + (1,)

        n0   = np.reshape(n0, result_shape)
        lmbd = np.broadcast_to(lmbd, shape).reshape(result_shape)
        mu   = np.broadcast_to(mu, shape).reshape(result_shape)

        x = x.reshape((1,) * len(shape) + (-1,))

        y = n0 * x ** mu * np.exp(- lmbd * x)
        return PSDData(x, y, self.size_parameter)
