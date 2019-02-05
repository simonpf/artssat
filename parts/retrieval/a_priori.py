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

    def get_covariance(self, data_provider, *args, **kwargs):
        if self.diagonal.size == 1:
            z = data_provider.get_altitude(*args, **kwargs)
            return sp.sparse.diags(self.diagonal[0] * np.ones(z.size),
                                   format = "coo")
        else:
            return self.diagonal

class SpatialCorrelation:
    def __init__(self,
                 covariance,
                 correlation_length,
                 correlation_type = "exp",
                 cutoff = 1e-2):
        self.covariance         = covariance
        self.correlation_length = correlation_length
        self.cutoff = cutoff

    def get_covariance(self, data_provider, *args, **kwargs):

        z = data_provider.get_altitude(*args, **kwargs)
        dz = np.abs(z.reshape(-1, 1) - z.reshape(1, -1))

        if self.correlation_type == "exp":
            corr = np.exp(- np.abs(dz / self.cl) )
        elif self.correlation_type == "gauss":
            corr = np.exp(- (dz / self.cl) ** 2)

        inds = corr < self.cutoff
        corr[inds] = 0.0

        covmat = self.covariance(data_provider, *args, **kwargs)
        if isinstance(covmat, sp.sparse.spamtrix):
            covmat = covmat.todense()

        return corr @ covmat

class Thikhonov:
    def __init__(self,
                 scaling    = 1.0,
                 mask       = None,
                 mask_value = 1e12):
        self.scaling    = scaling
        self.mask       = mask
        self.mask_value = mask_value

    def get_precision(data_provider, *args, **kwargs):

        z = data_provider.get_altitude(*args, **kwargs)
        t = data_provider.get_temperature(*args, **kwargs)
        n = z.size

        du2 = np.ones(n - 2)

        du1 = -4.0 * np.ones(n - 1)
        du1[0]  = -2.0
        du1[-1] = -2.0

        d     = 6.0 * np.ones(n)
        d[:2]  = [1, 5]
        d[-2:] = [5, 1]

        dd = np.zeros(n)
        if not self.height_mask is None:
            dd = self.height_mask(dd, 1e12, z)
        if not self.temperature_mask is None:
            dd = self.temperature_mask(dd, 1e12, t)
        d += dd

        precmat = sp.sparse.diags(diagonals = [du2, du1, d, du1, du2],
                                  offsets   = [2, 1, 0, -1, -2],
                                  format    = "coo")
        precmat *= self.scaling

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

        if hasattr("get_covariance", covariance):
            covariance_name = "get_" + name + "_covariance"
            self.__dict__[covariance_name] = self.get_covariance
        if hasattr("get_precision", precision):
            precision_name = "get_" + name + "_precision"
            self.__dict__[precision_name] = self.get_precision

        self.name = name
        self.covariance = covariance
        self.spatial_correlation = spatial_correlation

    def get_covariance(self, *args, *kwargs):
        self.covariance.get_covariance(self.owner, *args, **kwargs)

    def get_precision(self, *args, *kwargs):
        self.covariance.get_precision(self.owner, *args, **kwargs)


################################################################################
# DataProviderAPriori
################################################################################

class DataProviderAPriori(AprioriProviderBase):
    """
    A priori provider that propagates an atmospheric quantity :code:`name`
    as a priori mean profile from the data provider.
    """

    def __init__(self,
                 name,
                 covariance,
                 spatial_correlation = None):
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
        xa_name = "get_" + name + "_xa"
        self.__dict__[xa_name] = self.get_xa

    def get_xa(self, *args, **kwargs):

        f_name = "get_" + self.name
        try:
            f = getattr(self.owner, f_name)
        except:
            raise Expetion("DataProviderApriori instance requires get method "
                           " {0} from its owning data provider.")

        x = f(*args, **kwargs)
        return x

################################################################################
# Temperature mask
################################################################################

def tropopause_mask(t):
    """
    Returns a mask that is true only below the approximate height of the
    troposphere. The troposphere is detected as the first grid point
    where the lapse rate is negative and the temperature below 220.

    Arguments:

        t(code:`numpy.array`): 1D temperature array into which to detect
            the troposphere.
    """
    t_avg = 0.5 * (t[1:] + t[:-1])
    lr = - np.diff(t)
    i = np.where(np.logical_and(lr < 0, t_avg < 220))[0][0]
    inds = np.ones(t.size, dtype = np.bool)
    inds[i:inds.size] = False
    return inds

class TemperatureMask:
    """
    The temperature mask replaces values at grid points outside of the
    given temperature interval with another value.
    """
    def __init__(self, lower_limit, upper_limit, transition = 0):
        """
        Arguments:

            lower_limit(:code:`float`): The lower temperature limit

            upper_limit(:code:`float`): The upper temperature limit

            transition(:code:`int`): Length of linear transition between
                the original and the replacement values.
        """
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.transition = transition

    def __call__(self, x, xr, t):
        inds = np.logical_and(t.ravel() >= self.lower_limit,
                              t.ravel() <  self.upper_limit)
        inds = np.logical_and(tropopause_mask(t),
                              inds)

        x_new = np.copy(x)
        x_new[np.logical_not(inds)] = xr

        if self.transition > 1:
            k = np.ones(self.transition) / self.transition
            k2 = (self.transition - 1) // 2
            x_new[k2 : -k2] = np.convolve(x_new, k, mode = "valid")

        return x_new

    def apply_matrix(self, x, xr, t):
        """
        Applies mask to matrix. It is assumed that rows and columns
        of the matrix :code:`x` correspond to the temperature in
        `t`.

        Arguments:

            x(:code:`numpy.array`):

            xr(:code: `numpy.array`): The value to replaced the values
               for which the mask is :code:`False` with.

            t(:code: `numpy.array`): The temperature array corresponding
                to rows and column of :code:`x`.
        """
        x = np.copy(x)
        inds = np.logical_and(t.ravel() >= self.lower_limit,
                              t.ravel() <  self.upper_limit)
        inds = np.logical_and(tropopause_mask(t),
                              inds)
        inds     = np.logical_not(inds)
        inds2    = np.logical_or(inds.reshape(-1, 1),
                                 inds.reshape(1, -1))
        x[inds2] = 0.0
        x[inds, inds]  = xr
        return x

    def apply_matrix_off_diagonal(self, x, xr, t1, t2):
        """
        Applies mask to matrix where rows and column correspond
        to different temperatures.

        Arguments:

            x(:code:`numpy.array`):

            xr(:code: `numpy.array`): The value to replaced the values
               for which the mask is :code:`False` with.

            t1(:code: `numpy.array`): The temperature array corresponding
                to rows of :code:`x`.

            t2(:code: `numpy.array`): The temperature array corresponding
                to column of :code:`x`.
        """
        x = np.copy(x)
        inds1 = np.logical_and(t1.ravel() >= self.lower_limit,
                               t1.ravel() <  self.upper_limit)
        inds2 = np.logical_and(t2.ravel() >= self.lower_limit,
                               t2.ravel() <  self.upper_limit)
        inds12 = np.logical_and(inds1, inds2)
        inds12 = np.logical_and(tropopause_mask(t),
                                inds12)
        inds     = np.logical_not(inds12)
        x[inds]  = 0.0
        return x


################################################################################
# Fixed a priori
################################################################################

class FixedApriori(AprioriBase):
    """
    Returns a fixed a priori profile.
    
    """

    def __init__(self,
                 name,
                 xa,
                 covariance,
                 xr = 1e-12,
                 height_mask = None,
                 temperature_mask = None,
                 spatial_correlation = None):

        xa_name = "get_" + name + "_xa"
        self.__dict__[xa_name] = self.get_xa
        covariance_name = "get_" + name + "_covariance"
        self.__dict__[covariance_name] = self.get_covariance

        super().__init__(name)
        self.xa  = xa
        self.xr = xr
        self.covariance = covariance
        self.height_mask = height_mask
        self.temperature_mask = temperature_mask
        self.spatial_correlation = spatial_correlation

    def get_xa(self, *args, **kwargs):

        z = self.owner.get_altitude(*args, **kwargs)
        t = self.owner.get_temperature(*args, **kwargs)
        x = self.xa * np.ones(z.shape)

        if not self.height_mask is None:
            x = self.height_mask(x, self.xr, z)

        if not self.temperature_mask is None:
            x = self.temperature_mask(x, self.xr, t)

        return x

    def get_covariance(self, *args, **kwargs):

        z = self.owner.get_altitude(*args, **kwargs)
        t = self.owner.get_temperature(*args, **kwargs)
        covmat = np.diag(self.covariance * np.ones(z.size))

        if not self.spatial_correlation is None:
            covmat = self.spatial_correlation(covmat, z)

        if not self.height_mask is None:
            covmat = self.height_mask.apply_matrix(covmat, 1e-12, z)

        if not self.temperature_mask is None:
            covmat = self.temperature_mask.apply_matrix(covmat, 1e-12, t)

        return covmat
