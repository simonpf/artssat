"""
parts.retrieval.a_priori
------------------------

The :code:`retrieval.a_priori` sub-module provides modular data provider
object that can be used to build a priori data providers.
"""
from parts.data_provider import DataProviderBase
from parts.sensor import ActiveSensor, PassiveSensor
from parts.jacobian import Transformation
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
        self.diagonal   = diagonal
        self.mask       = mask
        self.mask_value = mask_value
        self.z_scaling  = z_scaling

    def get_covariance(self, data_provider, *args, **kwargs):
        precmat = self.get_precision(data_provider, *args, **kwargs)
        diag = precmat.diagonal()
        return sp.sparse.diags(1.0 / diag, format = "coo")

    def get_precision(self, data_provider, *args, **kwargs):

        z = data_provider.get_altitude(*args, **kwargs)
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
            mask = np.logical_not(self.mask(data_provider, *args, **kwargs))
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

        if hasattr(covariance, "get_covariance"):
            covariance_name = "get_" + name + "_covariance"
            self.__dict__[covariance_name] = self.get_covariance
        if hasattr(covariance, "get_precision"):
            precision_name = "get_" + name + "_precision"
            self.__dict__[precision_name] = self.get_precision

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
        """
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    def __call__(self, data_provider, *args, **kwargs):
        t    = data_provider.get_temperature(*args, **kwargs)
        inds = np.logical_and(t.ravel() >= self.lower_limit,
                              t.ravel() <  self.upper_limit)
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
        super().__init__(name, covariance)
        self._xa   = np.array(xa)
        self.mask = mask
        self.mask_value = mask_value

    def get_xa(self, *args, **kwargs):

        if self._xa.size == 1:
            z = self.owner.get_altitude(*args, **kwargs)
            xa = self._xa.ravel() * np.ones(z.size)
        else:
            xa = self._xa

        if not self.mask is None:
            mask = np.logical_not(self.mask(self.owner, *args, **kwargs))
            xa[mask] = self.mask_value

        return xa

################################################################################
# Functional a priori
################################################################################

class FunctionalAPriori(APrioriProviderBase):
    """
    Returns an a priori profile that does not depend on the atmospheric
    state.
    """
    def __init__(self,
                 name,
                 variable,
                 f,
                 covariance,
                 mask = None,
                 mask_value = 1e-12):
        super().__init__(name, covariance)
        self.variable   = variable
        self.f          = f
        self.mask       = mask
        self.mask_value = mask_value

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

            sensors(list of :code:`parts.sensor.Sensor`): Sensors used
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

    def setup(self, ws, data_provider, *args, **kwargs):
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

        ws.jacobianSetAffineTransformation(transformation_matrix = A, offset_vector = b)

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
        print(old_grid.shape, y.shape)
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
        print(old_grid.shape, new_grid.shape, y.shape)
        if self.quantity == "pressure":
            f = sp.interpolate.interp2d(old_grid[::-1], old_grid[::-1], y[::-1, ::-1],
                                        bounds_error = False, fill_value = np.nan)
            yi = f(new_grid[::-1], new_grid[::-1])[::-1, ::-1]
        else:
            f = sp.interpolate.interp2d(old_grid, old_grid, y,
                                        bounds_error = False)
            yi = f(new_grid, new_grid)
        return yi

    def get_xa(self, *args, **kwargs):
        self.a_priori.owner = self.owner
        xa = self.a_priori.get_xa(*args, **kwargs)
        return self._interpolate(xa, *args, **kwargs)

    def get_covariance(self, *args, **kwargs):
        if self._covariance is None:
            self.a_priori.owner = self.owner
            covmat = self.a_priori.get_covariance(*args, **kwargs)
            if isinstance(covmat, sp.sparse.spmatrix):
                covmat = covmat.todense()
            return self._interpolate_matrix(covmat, *args, **kwargs)
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
            return self._covariance.get_precision(self.owner, *args **kwargs)

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

        super().__init__(a_priori, None, quantity, covariance,
                         provide_retrieval_grid = provide_retrieval_grid)
        self.n_points = n_points
        self.mask = mask

        retrieval_p_name = "get_" + a_priori.name + "_p_grid"
        if provide_retrieval_grid:
            self.__dict__[retrieval_p_name] = self.get_retrieval_p_grid

        self.transition = transition

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
        i_first = np.where(mask)[0][0]
        i_last = np.where(mask)[0][-1]

        if i_first > 0:
            left = 1
        else:
            left = 0

        right = left + self.n_points

        if i_last < mask.size - 1:
            n = right + 1
        else:
            n = right

        new_grid = np.zeros(n)
        new_grid[left : right] = np.linspace(old_grid[i_first], old_grid[i_last], self.n_points)
        if left > 0:
            if self.transition is None:
                new_grid[0] = old_grid[i_first - 1]
            else:
                new_grid[0] = new_grid[1] - self.transition

        if right < n:
            if self.transition is None:
                new_grid[-1] = old_grid[i_last + 1]
            else:
                new_grid[-1] = new_grid[-2] + self.transition

        return old_grid, new_grid
