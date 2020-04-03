"""
artssat.retrieval.a_priori
--------------------------

The :code:`retrieval.a_priori` sub-module provides modular data provider
object that can be used to build a priori data providers.
"""
from artssat.data_provider import DataProviderBase
from artssat.sensor import ActiveSensor, PassiveSensor
from artssat.jacobian import Transformation
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
            mask = np.logical_not(self.mask(data_provider, *args, **kwargs))
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
                 cutoff = 1e-12,
                 mask = None,
                 mask_value = 1e-12,
                 z = None):
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
        self.mask = mask
        self.mask_value = mask_value
        self.z = z

    def get_covariance(self, data_provider, *args, **kwargs):

        if self.z is None:
            z = data_provider.get_altitude(*args, **kwargs)
        else:
            z = self.z
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
        diag = np.sqrt(covmat.diagonal())
        covmat = corr * np.array((diag.T * diag))

        if not self.mask is None:
            inds = np.logical_not(self.mask(data_provider, *args, **kwargs))
            inds2 = np.logical_or(inds.reshape(-1, 1), inds.reshape(1, -1))
            covmat[inds2] = 0.0
            covmat[inds, inds] = self.mask_value

        return np.asarray(covmat)

class Thikhonov:
    """
    Thikhonov regularization using second order finite differences.
    """
    def __init__(self,
                 scaling    = 1.0,
                 diagonal   = 0.0,
                 mask       = None,
                 mask_value = 1e12,
                 z_scaling  = True,
                 z_grid = None):
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
        self.diagonal   = diagonal
        self.mask       = mask
        self.mask_value = mask_value
        self.z_scaling  = z_scaling
        self.z_grid = z_grid

    def get_covariance(self, data_provider, *args, **kwargs):
        precmat = self.get_precision(data_provider, *args, **kwargs)
        diag = precmat.diagonal()
        return sp.sparse.diags(1.0 / diag, format = "coo")

    def get_precision(self, data_provider, *args, **kwargs):

        if self.z_grid is None:
            z = data_provider.get_altitude(*args, **kwargs)
            z_old = None
        else:
            z = self.z_grid
            z_old = data_provider.get_altitude(*args, **kwargs)
        n = z.size

        du2 = np.ones(n - 2)

        du1 = -4.0 * np.ones(n - 1)
        du1[0]  = -2.0
        du1[-1] = -2.0

        dl1 = np.copy(du1)
        dl2 = np.copy(du2)

        d     = 6.0 * np.ones(n)
        d[:2]  = [1, 5]
        d[-2:] = [5, 1]

        if self.diagonal > 0.0:
            d += self.diagonal

        if not self.mask is None:
            mask = self.mask(data_provider, *args, **kwargs).astype(np.float)
            if not z_old is None:
                f = sp.interpolate.interp1d(z_old, mask,
                                            axis = 0,
                                            bounds_error = False,
                                            fill_value = (mask[0], mask[-1]))
                mask = f(z) > 0.5

            mask = np.logical_not(mask)
            du1[mask[:-1]] = 0
            du2[mask[:-2]] = 0
            dl1[mask[1:]]  = 0
            dl2[mask[2:]]  = 0
            d[mask] = self.mask_value


        precmat = sp.sparse.diags(diagonals = [du2, du1, d, dl1, dl2],
                                  offsets   = [2, 1, 0, -1, -2],
                                  format    = "coo")
        precmat *= self.scaling

        if self.z_scaling:
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

        if hasattr(self, "get_x0"):
            x0_name = "get_" + name + "_x0"
            self.__dict__[x0_name] = self.get_x0

        if hasattr(covariance, "get_covariance"):
            covariance_name = "get_" + name + "_covariance"
            self.__dict__[covariance_name] = self.get_covariance
        if hasattr(covariance, "get_precision"):
            precision_name = "get_" + name + "_precision"
            self.__dict__[precision_name] = self.get_precision

        if hasattr(self, "get_mask"):
            precision_name = "get_" + name + "_mask"
            self.__dict__[precision_name] = self.get_mask

        self.name = name
        self._covariance = covariance

    def get_covariance(self, *args, **kwargs):
        return self._covariance.get_covariance(self.owner, *args, **kwargs)

    def get_precision(self, *args, **kwargs):
        return self._covariance.get_precision(self.owner, *args, **kwargs)

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

        tp = np.where(np.logical_and(lr < 0, t_avg < 220))[0]
        inds  = np.ones(t.size, dtype = np.bool)
        if len(tp > 0):
            i     = np.where(np.logical_and(lr < 0, t_avg < 220))[0][0]
            inds[i + 1 : inds.size] = False
        return inds

class FreezingLevel:
    def __init__(self,
                 lower_inclusive = False,
                 invert = False):
        self.lower_inclusive = lower_inclusive
        self.invert = invert

    def __call__(self, data_provider, *args, **kwargs):
        """
        Arguments:

            data_provider: Data provider describing the atmospheric scenario.

            *args: Arguments to forward to data provider.

            **kwargs: Keyword arguments to forward to data_provider.
        """
        t     = data_provider.get_temperature(*args, **kwargs)
        inds = np.where(t < 273.15)[0]
        if len(inds) > 0:
            i = inds[0]
        else:
            i = 0
        if self.lower_inclusive:
            i = max(i - 1, 0)
        inds = np.zeros(t.size, dtype = np.bool)
        inds[i:] = True

        if self.invert:
            inds = np.logical_not(inds)

        return inds

class TemperatureMask:
    """
    The temperature mask replaces values at grid points outside of the
    given temperature interval with another value.
    """
    def __init__(self,
                 lower_limit,
                 upper_limit,
                 lower_inclusive = False,
                 upper_inclusive = False):
        """
        Arguments:

            lower_limit(:code:`float`): The lower temperature limit
            upper_limit(:code:`float`): The upper temperature limit
            lower_inclusive: Whether or not to include highest grid
                point (in altitude) that does not lie in the half-open
                temperature interval between lower_limit and upper limit.
            lower_inclusive: Whether or not to include lowest grid
                point (in altitude) that does not lie in the half-open
                temperature interval between lower_limit and upper limit.

                point on lower limit.
            upper_inclusive: Whether or not to include adjacent grid
                point on upper limit.
        """
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.upper_inclusive = upper_inclusive
        self.lower_inclusive = lower_inclusive

    def __call__(self, data_provider, *args, **kwargs):
        t    = data_provider.get_temperature(*args, **kwargs)
        inds = np.logical_and(t.ravel() >= self.lower_limit,
                              t.ravel() <  self.upper_limit)
        if self.upper_inclusive:
            inds[1:] += inds[:-1]
        if self.lower_inclusive:
            inds[:-1] += inds[1:]
        return inds

class AltitudeMask:
    """
    The altitude mask replaces values at grid points outside of the
    given altitude interval with another value.
    """
    def __init__(self, lower_limit, upper_limit):
        """
        Arguments:

            lower_limit(:code:`float`): The lower altitude limit

            upper_limit(:code:`float`): The upper altitude limit

        """
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    def __call__(self, data_provider, *args, **kwargs):
        z    = data_provider.get_altitude(*args, **kwargs)
        inds = np.logical_and(z.ravel() >= self.lower_limit,
                              z.ravel() <  self.upper_limit)
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

        if not mask is None:
            if not hasattr(self, "get_mask"):
                self.__dict__["get_mask"] = self._get_mask

        super().__init__(name, covariance)
        self._xa   = np.array(xa)
        self.mask = mask
        self.mask_value = mask_value

    def _get_mask(self, data_provider, *args, **kwargs):
        if self.mask is None:
            xa = self._get_xa(data_provider, *args, **kwargs)
            mask = np.ones(xa.shape, dtype=np.bool)
        else:
            mask = self.mask(data_provider, *args, **kwargs)
        return mask

    def _get_xa(self, data_provider, *args, **kwargs):

        if self._xa.size == 1:
            z = data_provider.get_altitude(*args, **kwargs)
            xa = self._xa.ravel() * np.ones(z.size)
        else:
            xa = self._xa

        if not self.mask is None:
            mask = np.logical_not(self._get_mask(self, data_provider, *args, **kwargs))
            xa[mask] = self.mask_value

        return xa

    def get_xa(self, *args, **kwargs):
        return self._get_xa(self.owner, *args, **kwargs)

################################################################################
# Functional a priori
################################################################################

class FunctionalAPriori(APrioriProviderBase):
    """
    Returns an a priori profile that is a functional  transform
    of some variable.
    """
    def __init__(self,
                 name,
                 variable,
                 f,
                 covariance,
                 mask = None,
                 mask_value = 1e-12):

        if not mask is None:
            self.__dict__["get_mask"] = self._get_mask

        super().__init__(name, covariance)
        self.variable   = variable
        self.f          = f
        self.mask       = mask
        self.mask_value = mask_value

    def _get_mask(self, data_provider, *args, **kwargs):
        mask = self.mask(data_provider, *args, **kwargs)
        return mask.astype(np.float)

    def get_xa(self, *args, **kwargs):

        try:
            f_get = getattr(self.owner, "get_" + self.variable)
            x = f_get(*args, **kwargs)
        except:
            raise Exception("Could not get variable {} from data provider."
                            .format(self.variable))

        xa = self.f(x)

        if not self.mask is None:
            mask = np.logical_not(self.mask(self.owner, *args, **kwargs))
            xa[mask] = self.mask_value

        return xa

################################################################################
# Reference A Priori
################################################################################

class ReferenceAPriori(APrioriProviderBase):
    """
    Forwards value from data provider as a priori state. Useful for
    testing of retrieval implementation.
    """
    def __init__(self,
                 name,
                 covariance,
                 mask = None,
                 mask_value = 1e-12,
                 a_priori = None,
                 transformation = None,
                 variable = None):
        super().__init__(name, covariance)
        self.mask       = mask
        self.mask_value = mask_value
        self.a_priori = a_priori
        self.transformation = transformation

        if not variable is None:
            self.variable = variable
        else:
            self.variable = name

    def get_xa(self, *args, **kwargs):
        if not self.a_priori is None:
            self.a_priori.owner = self.owner
            return self.a_priori.get_xa(*args, **kwargs)
        else:
            f_get = getattr(self.owner, "get_" + self.variable)
            x = f_get(*args, **kwargs)
            if not self.transformation is None:
                x = self.transformation(x)
            return x

    def get_x0(self, *args, **kwargs):

        f_get = getattr(self.owner, "get_" + self.variable)
        x = f_get(*args, **kwargs)
        if not self.transformation is None:
            x = self.transformation(x)

        if not self.a_priori is None:
            if hasattr(self.a_priori, "mask") and not self.a_priori.mask is None:
                mask = np.logical_not(self.a_priori.mask(self.owner, *args, **kwargs))
                x[mask] = self.a_priori.mask_value

        return x

################################################################################
# Sensor a priori
################################################################################

class SensorNoiseAPriori(DataProviderBase):
    """
    Measurement error due to sensor noise.

    The :code:`SensorNoiseAPriori` class constructs a combined
    observation error covariance matrix from the noise characteristics
    of a list of sensors.

    The noise of particular sensors can be amplified by adding the
    scaling factor to the :code:`noise_scaling` attribute of the
    class.

    Attributes:

        noise_scaling(:code:`dict`): Dictionary mapping sensor names to
            noise scaling factors.
    """
    def __init__(self,
                 sensors):
        """
        Arguments:

            sensors(list of :code:`artssat.sensor.Sensor`): Sensors used
                in the observation for which to construct the observation
                error covariance matrix.

        """
        self.sensors = sensors
        self.noise_scaling = {}

    def get_observation_error_covariance(self, *args, **kwargs):
        m = 0

        stds = []

        for s in self.sensors:
            if isinstance(s, ActiveSensor):
                if s.name in self.noise_scaling:
                    c = self.noise_scaling[s.name]
                else:
                    c = 1.0
                stds += [c * s.nedt]

        for s in self.sensors:
            if isinstance(s, PassiveSensor):
                if s.name in self.noise_scaling:
                    c = self.noise_scaling[s.name]
                else:
                    c = 1.0
                stds += [c * s.nedt]

        sig = np.concatenate(stds).ravel()
        return sp.sparse.diags(sig ** 2.0, format = "coo")

################################################################################
# Reduced retrieval grids
################################################################################

class PiecewiseLinear(Transformation):
    def __init__(self, grid):
        self.grid = grid

    def initialize(self, data_provider, *args, **kwargs):
        old_grid, new_grid = self.grid._get_grids(*args, **kwargs)
        m = new_grid.size
        n = old_grid.size
        A = np.zeros((m, n))

        if m == 3:
            zl = new_grid[0]
            zr = new_grid[-1]
            A[0, old_grid < zl] = 1.0
            A[1, np.logical_and(old_grid >= zl, old_grid < zr)] = 1.0
            A[2, old_grid >= zr] = 1.0
        else:

            z = new_grid[0]
            zr = new_grid[1]
            conditions = [old_grid <= z, old_grid > z]
            values = [1.0, lambda x: np.maximum(1.0 - (x - z) / (zr - z), 0.0)]
            A[0, :] = np.piecewise(old_grid, conditions, values)

            for i in range(1, new_grid.size - 1):
                zl = new_grid[i - 1]
                z = new_grid[i]
                zr = new_grid[i + 1]
                conditions = [old_grid < z,
                            old_grid == z,
                            old_grid > z]
                values = [lambda x: np.maximum(1.0 - (z - x) / (z - zl), 0.0),
                        1.0,
                        lambda x: np.maximum(1.0 - (x - z) / (zr - z), 0.0)]
                A[i, :] = np.piecewise(old_grid, conditions, values)

            z = new_grid[-1]
            zl = new_grid[-2]
            conditions = [old_grid < z, old_grid >= z]
            values = [lambda x: np.maximum(1.0 - (z - x) / (z - zl), 0.0), 1.0]
            A[-1, :] = np.piecewise(old_grid, conditions, values)

        b = np.zeros(n)

        self.A = A
        self.b = b

    def setup(self, ws, data_provider, *args, **kwargs):
        self.initialize(data_provider, *args, **kwargs)
        ws.jacobianSetAffineTransformation(transformation_matrix = self.A,
                                           offset_vector = self.b)

    def __call__(self, x):
        A_ = A / np.sum(A, axis = -1, keepdims = True)
        return A_ @ (x - self.b)

    def invert(self, x):
        return self.A.T @ x + self.b


class ReducedVerticalGrid(APrioriProviderBase):

    def __init__(self,
                 a_priori,
                 grid,
                 quantity = "pressure",
                 covariance = None,
                 provide_retrieval_grid = True):

        if hasattr(a_priori, "mask"):
            if not a_priori.mask is None:
                if not hasattr(self, "get_mask"):
                    self.__dict__["get_mask"] = self._get_mask

        if covariance is None:
            super().__init__(a_priori.name, a_priori)
        else:
            super().__init__(a_priori.name, covariance)
        self.a_priori = a_priori
        self.new_grid = grid
        self.quantity = quantity
        self._covariance = covariance

        retrieval_p_name = "get_" + a_priori.name + "_p_grid"
        if provide_retrieval_grid:
            self.__dict__[retrieval_p_name] = self.get_retrieval_p_grid

    def _get_grids(self, *args, **kwargs):
        f_name = "get_" + self.quantity
        try:
            old_grid = getattr(self.owner, f_name)(*args, **kwargs)
        except:
            raise Exception("Data provider does not provide get function "
                            "for quantity {} required to determine original "
                            "size of retrieval grid."
                            .format(self.quantity))
        return old_grid, self.new_grid


    def _interpolate(self, y, *args, **kwargs):
        old_grid, new_grid = self._get_grids(*args, **kwargs)
        if self.quantity == "pressure":
            f = sp.interpolate.interp1d(old_grid[::-1], y[::-1],
                                        axis = 0,
                                        bounds_error = False,
                                        fill_value = (y[-1], y[0]))
            yi = f(new_grid[::-1])[::-1]
        else:
            f = sp.interpolate.interp1d(old_grid, y,
                                        axis = 0,
                                        bounds_error = False,
                                        fill_value = (y[0], y[-1]))
            yi = f(new_grid)
        return yi

    def _interpolate_matrix(self, y, *args, **kwargs):
        old_grid, new_grid = self._get_grids(*args, **kwargs)
        if self.quantity == "pressure":
            f = sp.interpolate.interp2d(old_grid[::-1], old_grid[::-1], y[::-1, ::-1],
                                        bounds_error = False, fill_value = np.nan)
            yi = f(new_grid[::-1], new_grid[::-1])[::-1, ::-1]
        else:
            f = sp.interpolate.interp2d(old_grid, old_grid, y,
                                        bounds_error = False)
            yi = f(new_grid, new_grid)

        if (yi.shape[0] > 1) and (yi.shape[1] > 1):
            # Check for degeneracy at boundaries.
            if np.all(yi[0, :] == yi[1, :]):
                yi[0, 0] = yi[1, 1]
                yi[0, 1:] = 0.0
                yi[1:, 0] = 0.0

            # Check for degeneracy at boundaries.
            if np.all(yi[-1, :] == yi[-2, :]):
                yi[-1, -1] = yi[-2, -2]
                yi[-1, :-1] = 0.0
                yi[:-1, -1] = 0.0
        return yi

    def _get_mask(self, data_provider, *args, **kwargs):
        mask = self.a_priori._get_mask(data_provider, *args, **kwargs)

        mask_i = self._interpolate(mask, *args, **kwargs)
        mask_i[mask_i > 0.0] = 1.0

        # Return if mask is false everywhere
        if np.all(np.logical_not(mask)):
            return mask_i


        old_grid, new_grid = self._get_grids(*args, **kwargs)
        i_0_old, i_1_old = np.where(mask)[0][[0, -1]]
        i_0_new, i_1_new = np.where(mask_i)[0][[0, -1]]

        if self.quantity == "pressure":
            if old_grid[i_0_old] > new_grid[i_0_new]:
                mask_i[max(i_0_new - 1, 0)] = 1.0
            if old_grid[i_1_old] < new_grid[i_1_new]:
                mask_i[min(i_1_new + 1, new_grid.size - 1)] = 1.0
        else:
            if old_grid[i_0_old] < new_grid[i_0_new]:
                mask_i[max(i_0_new - 1, 0)] = 1.0
            if old_grid[i_1_old] > new_grid[i_1_new]:
                mask_i[min(i_1_new + 1, new_grid.size - 1)] = 1.0

        return mask_i.astype(np.bool)

    def get_xa(self, *args, **kwargs):
        self.a_priori.owner = self.owner
        xa = self.a_priori.get_xa(*args, **kwargs)
        return self._interpolate(xa, *args, **kwargs)

    def get_x0(self, *args, **kwargs):
        self.a_priori.owner = self.owner
        if hasattr(self.a_priori, "get_x0"):
            x0 = self.a_priori.get_x0(*args, **kwargs)
        else:
            x0 = self.a_priori.get_xa(*args, **kwargs)
        return self._interpolate(x0, *args, **kwargs)

    def get_covariance(self, *args, **kwargs):
        if self._covariance is None:
            self.a_priori.owner = self.owner
            covmat = self.a_priori.get_covariance(*args, **kwargs)
            if isinstance(covmat, sp.sparse.spmatrix):
                covmat = covmat.todense()
            covmat = self._interpolate_matrix(covmat, *args, **kwargs)
            mask = self._get_mask(self.owner, *args, **kwargs)
            for i in np.where(np.logical_not(mask))[0]:
                covmat[i, i + 1 :] = 0.0
                covmat[i+1 :, i] = 0.0
            return covmat
        else:
            return self._covariance.get_covariance(self.owner, *args, **kwargs)

    def get_precision(self, *args, **kwargs):
        if self._covariance is None:
            self.a_priori.owner = self.owner
            precmat = self.a_priori.get_precision(*args, **kwargs)
            if isinstance(precmat, sp.sparse.spmatrix):
                precmat = precmat.todense()
            return self._interpolate_matrix(precmat, *args, **kwargs)
        else:
            return self._covariance.get_precision(self.owner, *args, **kwargs)

    def get_retrieval_p_grid(self, *args, **kwargs):
        if self.quantity == "pressure":
            return self._get_grids(*args, **kwargs)[1]
        else:
            p = self.owner.get_pressure(*args, **kwargs)
            return self._interpolate(p, *args, **kwargs)


class MaskedRegularGrid(ReducedVerticalGrid):

    def __init__(self,
                 a_priori,
                 n_points,
                 mask,
                 quantity = "pressure",
                 covariance = None,
                 provide_retrieval_grid = True,
                 transition = None):

        if not a_priori.mask is None:
            if not hasattr(self, "get_mask"):
                self.__dict__["get_mask"] = self._get_mask

        super().__init__(a_priori, None, quantity, covariance,
                         provide_retrieval_grid = provide_retrieval_grid)
        self.n_points = n_points
        self.mask = mask

        retrieval_p_name = "get_" + a_priori.name + "_p_grid"
        if provide_retrieval_grid:
            self.__dict__[retrieval_p_name] = self.get_retrieval_p_grid

        self.transition = transition

    def _get_mask(self, data_provider, *args, **kwargs):
        mask = np.ones((self.n_points + 2,))
        mask[1] = 0.0
        mask[-1] = 0.0
        return mask

    def _get_grids(self, *args, **kwargs):
        mask = self.mask
        f_name = "get_" + self.quantity
        try:
            old_grid = getattr(self.owner, f_name)(*args, **kwargs)
        except:
            raise Exception("Data provider does not provide get function "
                            "for quantity {} required to determine original "
                            "size of retrieval grid."
                            .format(self.quantity))

        mask = self.mask(self.owner, *args, **kwargs)

        if len(np.where(mask)[0]) > 0:
            i_first = np.where(mask)[0][0]
            i_last = np.where(mask)[0][-1]
        else:
            i_first = 0
            i_last  = len(mask) - 1

        n = min(self.n_points + 2, mask.sum() + 2)
        new_grid = np.zeros((n,))
        new_grid[1 : -1] = np.linspace(old_grid[i_first], old_grid[i_last],
                                       n - 2)

        if n == 2:
            new_grid[0] = old_grid[0]
            new_grid[1] = old_grid[2]
            return old_grid, new_grid

        # Left
        if self.transition is None:
            if i_first > 0:
                new_grid[-1] = old_grid[i_first - 1]
            else:
                new_grid[0] = 2 * new_grid[1] - new_grid[2]
        else:
            new_grid[0] = new_grid[1] - self.transition

        # Right
        if self.transition is None:
            if i_last < old_grid.size - 1:
                new_grid[-1] = old_grid[i_last + 1]
            else:
                new_grid[-1] = 2.0 * new_grid[-2] - new_grid[-3]
        else:
            new_grid[-1] = new_grid[-2] + self.transition

        return old_grid, new_grid
