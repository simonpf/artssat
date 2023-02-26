r"""
The Field 07 snow PSD
=====================

This module implements the single-moment PSD for snow proposed
in:

Field, P. R., Heymsfield, A. J., & Bansemer, A. (2007). Snow Size
 Distribution Parameterization for Midlatitude and Tropical Ice Clouds
"""
import numpy as np

from pyarts.workspace import arts_agenda
from artssat.scattering.psd.arts.arts_psd import ArtsPSD, D_max
from artssat.scattering.psd.data.psd_data import PSDData


def estimate_moment(m2, n, temperature):
    """
    Parametrization to estimate PSD moments.

    Args:
        m2: Estimate of the second moment of the PSD
        n: The moment to estimate.
        temperature: The atmospheric temperature.

    Return:
        Estimate of n-th moment for given temperature and second moment
        of the PSD.
    """
    an = np.exp(13.6 - 7.76 * n + 0.479 * n**2)
    bn = -0.0361 + 0.01151 * n + 0.00149 * n**2
    cn = 0.807 + 0.00581 * n + 0.0457 * n**2
    return an * np.exp(bn * temperature) * m2**cn


def rescaled_psd_midlatitude(x):
    """
    The rescaled PSD shape for the mid-latitude regime.

    Args:
        x: The characteristic particle size.

    Return:
        The particle density normalized using the third and second PSD
        moments.
    """
    return 141.0 * np.exp(-16.8 * x) + 102.0 * x**2.07 * np.exp(-4.82 * x)


def rescaled_psd_tropical(x):
    """
    The rescaled PSD shape for the tropical regime.

    Args:
        x: The characteristic particle size.

    Return:
        The particle density normalized using the third and second PSD
        moments.
    """
    return 152.0 * np.exp(-12.4 * x) + 3.28 * x**-0.78 * np.exp(-1.94 * x)


class F07(ArtsPSD):
    """
    The Field07 PSD for snow particle.

    This single-moment PSD represents the PSD of ice hydrometeors using the
    ice water content, which is assumed to proportional to the second moment
    of the PSD, and the atmospheric temperature. The PSD distinguishes two
    regimes: 'tropical' for tropical anvil ice clouds and 'mid-latitude' for
    mid-latitude stratiform clouds.
    """
    @classmethod
    def from_psd_data(self, psd, a, b):
        r"""
        Create a F07 from given psd data.

        Args:

            psd(PSDData or other PSD): PSD data from which to create the MY05
            representation.
            a(:code:`float`): :math:`a` coefficient of the mass-size relationship
            b(:code:`float`): :math:`b` coefficient of the mass-size relationship
        """
        mass_density = psd.get_mass_density()
        return F07(mass_density, a, b)

    def __init__(self, mass_density=None, a=1.0, b=2.0, regime="tropical"):
        """
        Args:
            mass_density: The ice water content.
            a: Alpha parameter of the mass-size relationship
            b: Beta parameter of the mass-size relationship
            regime: One of the two optional regimes: 'tropical' or 'mid-latitude'.
        """
        self.mass_density = mass_density
        self.regime = regime.lower()
        super().__init__(D_max(a, b))
        self.t_max = 274.0

    def convert_from(self, psd):
        r"""
        Convert given psd to F07 PSD.

        Args:
            psd: The PSD to convert to F07.
        """
        self.mass_density = psd.get_mass_density()

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
        The ARTS WSM implementing the MY05 PSD.
        """
        if self.regime == "tropical":
            regime_str = "TR"
        else:
            regime_str = "ML"

        @arts_agenda
        def pnd_call(ws):
            ws.psdFieldEtAl07(
                regime=regime_str,
                t_min=self.t_min,
                t_max=self.t_max,
            )

        return pnd_call

    def get_moment(self, p, reference_size_parameter=None):
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

    def evaluate(self, x, temperature):
        r"""
        Compute a numeric representation of the PSD data.

        Parameters:

            x(:code:`numpy.ndarray`): Array containing the values of the size
            parameter at which to evaluate the PSD.

        Returns:

            :code:`PSDData` object containing the numeric PSD data corresponding
            to this PSD.

        """
        iwc = self.mass_density
        m2 = self.mass_density / self.size_parameter.a

        m3 = estimate_moment(m2, 3, temperature)

        dm = x * m2 / m3
        if self.regime == "tropical":
            phi = rescaled_psd_tropical(dm)
        else:
            phi = rescaled_psd_midlatitude(dm)

        y = phi * m2**4 / m3**3
        print(x)
        return PSDData(x, y, self.size_parameter)
