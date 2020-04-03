r"""
The modified gamma distribution PSD
===================================

The form of the modified gamma distribution (MGD) used here is as follows:

.. math::

    \frac{N(X)}{dX} = N \frac{\nu}{\Gamma(1 + \alpha)}\lambda^{\nu(1 + \alpha)}
                        D^{\nu(1 + \alpha) - 1} \cdot \exp \{-(\lambda D)^\nu\}.

The distribution is described by four parameters:

    1. The intercept parameter :math:`N`
    2. The slope parameter :math:`\lambda`
    3. The shape parameter :math:`\alpha`
    4. The parameter :math:`\nu`

"""
import numpy as np
import scipy as sp
from scipy.special import gamma
from artssat import dimensions as dim
from artssat.scattering.psd.arts.arts_psd import ArtsPSD
from artssat.scattering.psd.data.psd_data import PSDData, D_eq

class ModifiedGamma(ArtsPSD):
    r"""
    The :class:`ModifiedGamma` class describes the size distribution of
    scattering particles in an atmosphere using the four parameters of the
    particle size distribution.
    """

    properties = [("intercept_parameter", (dim.p, dim.lat, dim.lon), np.ndarray),
                  ("alpha", (dim.p, dim.lat, dim.lon), np.ndarray),
                  ("lmbd", (dim.p, dim.lat, dim.lon), np.ndarray),
                  ("nu", (dim.p, dim.lat, dim.lon), np.ndarray)]

    def __init__(self,
                 size_parameter,
                 intercept_parameter = None,
                 alpha = None,
                 lmbd = None,
                 nu = None):
        r"""
        Create instance of the modified gamma distribution with given parameters.

        If any of the parameters is neither provided and nor explicitly set
        afterwards, it will be requested from the data provider. However, most
        operations on PSDs will require the values to be set and can thus first
        be performed when the object has access to the data.

        Parameters:
            size_parameter(SizeParameter): The SizeParameter instance describing
            the size parameter that should be used fo PSD.

            intercept_parameter(numpy.float or ndarray): The intercept parameter
            :math:`N`

            alpha(numpy.float or ndarray): The shape parameter :math:`\alpha`.
                Must be broadcastable into the shape of N.

            lmbd(numpy.float or ndarray): The slope parameter :math:`\lambda`.
                Must be broadcastable into the shape of N.

            nu(numpy.float or ndarray): The :math:`\nu` parameter. Must be
                broadcastable into the shape of N.
        """
        if not intercept_parameter is None:
            self.intercept_parameter = intercept_parameter

        shape = self.intercept_parameter.shape

        if not alpha is None:
            try:
                self.alpha = np.broadcast_to(alpha, shape)
            except:
                raise Exception("Could not broadcast alpha parameter to shape "
                                "of intercept parameter.")

        if not lmbd is None:
            try:
                self.lmbd = np.broadcast_to(lmbd, shape)
            except:
                raise Exception("Could not broadcast lambda parameter to shape "
                                " of intercept parameter.")

        if not nu is None:
            try:
                self.nu = np.broadcast_to(nu, shape)
            except:
                raise Exception("Could not broadcast nu parameter to shape "
                                " of N parameter.")

        super().__init__(size_parameter)

    def _get_parameters(self):
        """
        Checks if parameters of the PSD are available and tries to broadcast
        them to the shape of the intercept parameter.

        Returns:

            :code:`tuple(n, alpha, lmbd, nu)` containing the four parameters of
            the PSD.

        Raises:

            An exception if any of the MGD parameters is not set or cannot be
            broadcasted.
        """

        n = self.intercept_parameter
        if n is None:
           raise Exception("The intercept parameter needs to be set to use"
                            " this function.")
        shape = n.shape

        # Lambda parameter

        lmbd = self.lmbd
        if lmbd is None:
            raise Exception("The lambda parameter needs to be set to use "
                            "this function")
        try:
            lmbd = np.broadcast_to(lmbd, shape)
        except:
            raise Exception("Could not broadcast lambda paramter to the shape"
                            "of the provided intercept parameter N.")

        # Alpha parameter

        alpha = self.alpha
        if alpha is None:
            raise Exception("The alpha parameter needs to be set to use "
                            "this function.")
        try:
            alpha = np.broadcast_to(alpha, shape)
        except:
            raise Exception("Could not broadcast alpha paramter to the shape"
                            "of the provided intercept parameter N.")

        # Nu parameter

        nu = self.nu
        if nu is None:
            raise Exception("The nu parameter needs to be set to use this"
                            "function.")

        try:
            nu = np.broadcast_to(nu, shape)
        except:
            raise Exception("Could not broadcast nu paramter to the shape"
                            "of the provided intercept parameter N.")

        return n, lmbd, alpha, nu

    @property
    def moment_names(self):
        r"""
        The free parameters of the PSD.
        """
        return []

    def get_moment(self, p, reference_size_parameter = None):
        r"""
        Computes the :math:`p` th moment :math:`M(p)` of the PSD using

        .. math::
            M(p) = \frac{N}{\lambda} \frac{\Gamma (1 + \alpha + p / \nu )}
            {\Gamma({1 + \alpha})}.

        Parameters:
            p(np.float): Which moment of the PSD to compute.

        Raises:
            Exception: If any of the parameters of the PSD is not set.

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

        n, lmbd, alpha, nu = self._get_parameters()

        m = n / lmbd ** p
        m *= gamma(1 + alpha + p / nu)
        m /= gamma(1 + alpha)

        return c * m

    def get_mass_density(self):
        r"""
        Computes the mass density :math: `\rho_m` for the given bulk elements
        using

        .. math::
            \rho_m = a \cdot M(b).

        where :math:`a` and :math:`b` are the coefficients of the mass-size
        relation of the size parameter.

        Returns:

            :code:`numpy.ndarray` containing the mass density corresponding
            to each volume element described by the PSD.
        """
        a = self.size_parameter.a
        b = self.size_parameter.b
        return a * self.get_moment(b)

    @property
    def pnd_call_agenda(self):
        r"""
        ARTS agenda that contains the call to the WSM that computed this PSD.
        """

        n0 = np.nan
        if not self.intercept_parameter is None \
           and self.intercept_parameter.size == 1:
            n0 = self.intercept_parameter[0]

        mu = np.nan
        if not self.mu is None \
           and self.mu.size == 1:
            mu = self.mu[0]

        lmbd = np.nan
        if not self.lmbd is None \
           and self.lmbd.size == 1:
            lmbd = self.lmbd[0]

        nu = np.nan
        if not self.nu is None \
           and self.nu.size == 1:
            nu = self.nu[0]

        @arts_agenda
        def pnd_call(ws):
            ws.psdMgd(n0 = n0,
                      mu = mu,
                      la = lambd,
                      gam   = nu,
                      t_min  = self.t_min,
                      t_max  = self.t_max)

        return pnd_call

    def evaluate(self, x):
        r"""
        Computes the values of this modified gamma distribution evaluated at
        the given size grid :code:`x`.

        Parameters:

            x(numpy.array): Array containing the values of the size parameter
            at which to evaluate the PSD.

        Returns:

            :class:`PSDData` object containing the numeric PSD data obtained
            by evaluating this PSD at the given values of the size parameter.

        """
        n, lmbd, alpha, nu = self._get_parameters()

        shape = n.shape

        n     = n.reshape(shape + (1,))
        lmbd  = lmbd.reshape(shape  + (1,))
        alpha = alpha.reshape(shape  + (1,))
        nu    = nu.reshape(shape + (1,))

        print(n, lmbd, alpha, nu)

        y = n * nu / gamma(1 + alpha)
        y *= lmbd ** (nu * (1.0 + alpha))
        y = y * x ** (nu * (1.0 + alpha) - 1) \
             * np.exp(- (lmbd * x) ** nu)
        return PSDData(x, y, self.size_parameter)
