r"""
The Milbrandt-Yau (2005) PSD
============================

The Milbrandt-Yau (2005) PSD used in the GEM model:

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
from parts import dimensions as dim
from parts.arts_object import ArtsObject
from parts.scattering.psd.modified_gamma     import ModifiedGamma
from parts.scattering.psd.data.psd_data      import D_max
from parts.scattering.psd.arts.arts_psd import ArtsPSD
from parts.scattering.psd.data.psd_data import PSDData

class MY05(ArtsPSD, metaclass = ArtsObject):
    r"""
    The :class:`MY05` class describes the size distributions of particles
    in an atmosphere using the number density :math:`\rho_n` and mass
    density :math:`\rho_m` as predictive moments.

    The :math:`nu` and :math:`mu` parameters take on fixed values depending
    on the hydrometero type.
    """

    properties = [("number_density", (dim.p, dim.lat, dim.lon), np.ndarray),
                  ("mass_density", (dim.p, dim.lat, dim.lon), np.ndarray),
                  ("nu", (), np.float),
                  ("mu", (), np.float)]

    def __init__(self, nu, mu, a, b,
                 number_density = None,
                 mass_desnity = None)
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
        self.nu = nu
        self.mu = mu

        if not number_density is None:
            self.number_density = number_density

        if not mass_density is None:
            self.mass_density = mass_density

        super().__init__(D_max(a, b))

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
               * gamma((nu + 1 + a) / mu) \
               / gamma((nu + 1) / mu)
        lmbd = lmbd ** (mu / b)

        n0 = n * mu * lmbd ** ((nu + 1.0) / mu) / gamma((nu + 1.0) / mu)

        return n0, lmbd, mu, nu

    @property
    def moment_names(self):
        """
        The names of the predictive moments of the PSD.
        """
        return ["number_density", "mass_density"]

    def pnd_call_agenda(self):
        """
        The ARTS WSM implementing the MY05 PSD.
        """
        pass

    def get_moment(self, p):
        r"""
        Analytically computes the :math:`p` th moment of the PSD using

        .. math::

            M(p) = \frac{N_0}{\mu} \lambda ^{-\frac{\nu + p + 1}{\mu}}
                   \Gamma ( \frac{\nu + p + 1}{\mu})

        Returns:

            :code:`numpy.ndarray` containing the :math:`p` th moment for
            all volume elements described by the PSD.

        """
        n0, lmbd, mu, nu = self._get_parameters()
        m = n0 / mu * lmbd ** (-(nu + p + 1) / mu) * gamma((nu + 1.0 + p) / mu)
        return m

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

        Returns:

            :code:`PSDData` object containing the numeric PSD data corresponding
            to this PSD.

        """
        n0, lmbd, mu, nu = self._get_parameters()
        y = n0 * x ** nu * np.exp(- lmbd * x ** mu)
        print(x)
        print(y)
        return PSDData(x, y, self.size_parameter)
