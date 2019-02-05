"""
parts.retrieval.a_priori
------------------------

The :code:`retrieval.a_priori` sub-module provides modular data provider
object that can be used to build a priori data providers.
"""
from parts.data_provider import DataProviderBase
import numpy as np
import scipy as sp
import scipy.sparse


################################################################################
# Covariances
################################################################################

class Diagonal:
    def __init__(self,
                 diagonal,
                 mask = None,
                 mask_value = 1e-12):
        self.diagonal = np.array(diagonal)
        self.mask = mask
        self.mask_value = mask_value

    def get_covariance(self, data_provider, *args, **kwargs):

        if not self.mask is None:
            mask = self.mask(data_provider, *args, **kwargs)
        else:
            mask = []

        if self.diagonal.size == 1:
            z = data_provider.get_altitude(*args, **kwargs)
            diagonal = self.diagonal * np.ones(z.size)
        else:
            diagonal = self.diagonal

        diagonal[mask] = self.mask_value
        diagonal = sp.sparse.diags(diagonal, format = "coo")
        return diagonal

class SpatialCorrelation:
    """
    Adds spatial correlation to a given covariance matrix.
    """
    def __init__(self,
                 covariance,
                 correlation_length,
                 correlation_type = "exp",
                 cutoff = 1e-2):
        """
        Arguments:

            covariance: Covariance object providing the original covariance
                matrix to which to apply the spatial correlation.

            correlation_length(:code:`float`): Correlation length in meters.

            correlation_type(:code:`str`): Type of the correlation to apply.

            cutoff(:code:`float`): Threshold below which to set correlation
                coefficients to zero.
        """
        self.covariance         = covariance
        self.correlation_length = correlation_length
        self.correlation_type   = correlation_type
        self.cutoff = cutoff

    def get_covariance(self, data_provider, *args, **kwargs):

        z = data_provider.get_altitude(*args, **kwargs)
        dz = np.abs(z.reshape(-1, 1) - z.reshape(1, -1))

        if self.correlation_type == "exp":
            corr = np.exp(- np.abs(dz / self.correlation_length) )
        elif self.correlation_type == "gauss":
            corr = np.exp(- (dz / self.correlation_length) ** 2)

        inds = corr < self.cutoff
        corr[inds] = 0.0

        covmat = self.covariance.get_covariance(data_provider, *args, **kwargs)
        if isinstance(covmat, sp.sparse.spmatrix):
            covmat = covmat.todense()

        return corr @ covmat

class Thikhonov:
    """
    Thikhonov regularization using second order finite differences.
    """
    def __init__(self,
                 scaling    = 1.0,
                 mask       = None,
                 mask_value = 1e12,
                 z_scaling  = True):
        """
        Arguments:
            scaling(:code:`np.float`): Scalar to scale the precision matrix with.

            mask: A mask object defining indices on the diagonal on the precision
                matrix to which to as :code:`mask_value`.

            mask_value(:code:`float`): Scalar to add to the diagonal of the precision
                matrix on locations defined by :code:`mask`.

            z_scaling(:code:`Bool`): Whether or not to scale matrix coefficients
                according to height differences between levels.
        """
        self.scaling    = scaling
        self.mask       = mask
        self.mask_value = mask_value
        self.z_scaling  = z_scaling

    def get_precision(self, data_provider, *args, **kwargs):

        z = data_provider.get_altitude(*args, **kwargs)
        n = z.size

        du2 = np.ones(n - 2)

        du1 = -4.0 * np.ones(n - 1)
        du1[0]  = -2.0
        du1[-1] = -2.0

        d     = 6.0 * np.ones(n)
        d[:2]  = [1, 5]
        d[-2:] = [5, 1]

        if not self.mask is None:
            mask = self.mask(data_provider, *args, **kwargs)
            d_mask = np.zeros(n)
            d_mask[mask] = self.mask_value
            d += d_mask

        precmat = sp.sparse.diags(diagonals = [du2, du1, d, du1, du2],
                                  offsets   = [2, 1, 0, -1, -2],
                                  format    = "coo")
        precmat *= self.scaling

        if self.z_scaling:
            z = data_provider.get_altitude(*args, **kwargs)
            zf = (np.diff(z) / np.diff(z).mean()) ** 2.0
            zf1 = np.zeros(z.shape)
            zf1[1:]  += zf
            zf1[:-1] += zf
            zf1[1:-1] *= 0.5
            precmat = sp.sparse.diags(diagonals = [zf1], offsets = [0], format = "coo") * precmat

        return precmat.tocoo()

################################################################################
# APrioriProviderBase
################################################################################

class APrioriProviderBase(DataProviderBase):
    def __init__(self,
                 name,
                 covariance):
        """
        Create :class:`DataProviderApriori` instance that will provide
        the value of the quantity :code:`name` from the owning data
        provider as a priori mean state.

        Arguments:

            name(:code:`name`): Name of the quantity that this a priori
                provider should propagate.

            covariance(:code:`numpy.array` or :code:`float`): Diagonal or
                full covariance matrix to be provided as covariance matrix
                for the retrieval quantity.

            spatial_correlation: Functor that is applied to the covariance
                matrix before this is returned.
        """

        super().__init__()
        xa_name = "get_" + name + "_xa"
        self.__dict__[xa_name] = self.get_xa

        covariance.provider = self

        if hasattr(covariance, "get_covariance"):
            covariance_name = "get_" + name + "_covariance"
            self.__dict__[covariance_name] = self.get_covariance
        if hasattr(covariance, "get_precision"):
            precision_name = "get_" + name + "_precision"
            self.__dict__[precision_name] = self.get_precision

        self.name = name
        self.covariance = covariance

    def get_covariance(self, *args, **kwargs):
        return self.covariance.get_covariance(self.owner, *args, **kwargs)

    def get_precision(self, *args, **kwargs):
        return self.covariance.get_precision(self.owner, *args, **kwargs)


################################################################################
# Masks
################################################################################

class And:
    """
    Creates a combined mask by applying logical and to a list
    of single masks.
    """
    def __init__(self, *args):
        self.masks = list(args)

    def __call__(self, data_provider, *args, **kwargs):
        """
        Arguments:

            data_provider: Data provider describing the atmospheric scenario.

            *args: Arguments to forward to data provider.

            **kwargs: Keyword arguments to forward to data_provider.
        """
        masks = []
        for m in self.masks:
            masks += [m(data_provider, *args, **kwargs)]

        m_and = masks[0]
        for m in masks[1:]:
            m_and = np.logical_and(m_and, m)

        return m_and

class TropopauseMask:
    """
    Returns a mask that is true only below the approximate height of the
    troposphere. The troposphere is detected as the first grid point
    where the lapse rate is negative and the temperature below 220.
    """
    def __init__(self):
        pass

    def __call__(self, data_provider, *args, **kwargs):
        """
        Arguments:

            data_provider: Data provider describing the atmospheric scenario.

            *args: Arguments to forward to data provider.

            **kwargs: Keyword arguments to forward to data_provider.
        """
        t     = data_provider.get_temperature(*args, **kwargs)
        t_avg = 0.5 * (t[1:] + t[:-1])
        lr    = - np.diff(t)
        i     = np.where(np.logical_and(lr < 0, t_avg < 220))[0][0]
        inds  = np.ones(t.size, dtype = np.bool)
        inds[i : inds.size] = False
        return inds

class TemperatureMask:
    """
    The temperature mask replaces values at grid points outside of the
    given temperature interval with another value.
    """
    def __init__(self, lower_limit, upper_limit):
        """
        Arguments:

            lower_limit(:code:`float`): The lower temperature limit

            upper_limit(:code:`float`): The upper temperature limit

            transition(:code:`int`): Length of linear transition between
                the original and the replacement values.
        """
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    def __call__(self, data_provider, *args, **kwargs):
        t    = data_provider.get_temperature()
        inds = np.logical_and(t.ravel() >= self.lower_limit,
                              t.ravel() <  self.upper_limit)
        return inds

################################################################################
# A priori providers
################################################################################

class DataProviderAPriori(APrioriProviderBase):
    """
    A priori provider that propagates an atmospheric quantity :code:`name`
    as a priori mean profile from the data provider.
    """

    def __init__(self,
                 name,
                 covariance):
        """
        Create :class:`DataProviderApriori` instance that will provide
        the value of the quantity :code:`name` from the owning data
        provider as a priori mean state.

        Arguments:

            name(:code:`name`): Name of the quantity that this a priori
                provider should propagate.

            covariance(:code:`numpy.array` or :code:`float`): Diagonal or
                full covariance matrix to be provided as covariance matrix
                for the retrieval quantity.

            spatial_correlation: Functor that is applied to the covariance
                matrix before this is returned.
        """

        super().__init__(name, covariance)

    def get_xa(self, *args, **kwargs):

        f_name = "get_" + self.name
        try:
            f = getattr(self.owner, f_name)
        except:
            raise Expetion("DataProviderApriori instance requires get method "
                           " {0} from its owning data provider.")

        x = f(*args, **kwargs)
        return x


class FixedAPriori(APrioriProviderBase):
    """
    Returns an a priori profile that does not depend on the atmospheric
    state.
    """
    def __init__(self,
                 name,
                 xa,
                 covariance,
                 mask = None,
                 mask_value = 1e-12):
        super().__init__(name, covariance)
        self.xa   = np.array(xa)
        self.mask = mask
        self.mask_value = mask_value

    def get_xa(self, *args, **kwargs):

        if self.xa.size == 1:
            z = self.owner.get_altitude(*args, **kwargs)
            xa = self.xa.ravel() * np.ones(z.size)
        else:
            xa = self.xa

        if not self.mask is None:
            mask = self.mask(self.owner, *args, **kwargs)
            xa[mask] = self.mask_value

        return xa

################################################################################
# Sensor a priori
################################################################################
