"""
artssat.jacobian
----------------

The :code:`jacobian` module handles calculations of Jacobians in ARTS.
Functionality for computing Jacobians in ARTS is implemented through
three classes:

1. :class:`JacobianCalculation` handles the actual calculation of
   Jacobians and coordinates the interaction with the ARTS workspace.

2. :class:`JacobianQuantity` defines the general interface for quantities
   for which a Jacobian can be calculated and how to toggle the calculation
   of the Jacobian for a given quantity.

3. :class:`Jacobian` handles quantity-specific settings and results. This
   class must be defined for each Jacobian quantity separately.

Calculating Jacobians
=====================

To calculate the Jacobian of a quantity :code:`q` it suffices
to add it to the `jacobian` of the :class:`ArtsSimulation`:

::

    simulation.jacobian.add(q)

This will add the quantity :code:`q` to the quantities for which a Jacobian
should be computed. Moreover it will instantiate the :class:`Jacobian` class
of the quantity :code:`q` and set the :code:`q.jacobian` property of :code:`q`.

It is important to note that :code:`simulation.jacobian` and :code:`q.jacobian`
are of different types: :code:`simulation.jacobian` is of type
:code:`JacobianCalculation` and handles the calculation of the Jacobian for
multiple quantities whereas :code:`q.jacobian` holds the settings and results
of the Jacobian calculation for :code:`q`.

Reference
=========
"""

import scipy as sp
import scipy.interpolate
import numpy as np

from abc import ABCMeta, abstractmethod, abstractproperty

from artssat.sensor      import ActiveSensor, PassiveSensor
from artssat.arts_object import ArtsObject, arts_property
from artssat.arts_object import Dimension as dim

################################################################################
# Transformations
################################################################################

class Transformation(metaclass = ABCMeta):
    """
    Abstract base class for transformations of Jacobian quantities.

    ARTS allows the calculation of certain transformations of Jacobian
    quantities. In artssat, these transformations are represented by subclasses
    of the :class:`Transformation` class, which defines the general interface
    for transformations of Jacobian quantities.

    The interface consists of two functions:

    - :code:`setup`: When the :code:`setup` function is called the
      transformation object is expected to call the appropriate workspace
      function so that the transformation is added to the most recently added
      JacobianQuantity.

    - :code:`__call__`: Transformation objects should be callable and calling
      them should apply the transformation to a given numeric argument.
    """
    def __init__(self):
        pass

    @abstractmethod
    def setup(self, ws, data_provider, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def invert(self, y):
        pass

class Log10(Transformation):
    """
    The decadal logarithm transformation $f(x) = \log_{10}(x)$.
    """
    def __init__(self, minimum = 1e-20):
        Transformation.__init__(self)
        self.minimum = minimum

    def setup(self, ws, data_provider, *args, **kwargs):
        ws.jacobianSetFuncTransformation(transformation_func = "log10")

    def __call__(self, x):
        return np.log10(np.maximum(x, self.minimum))

    def invert(self, y):
        return 10.0 ** y

class Log(Transformation):
    """
    The natural logarithm transformation $f(x) = \log_{10}(x)$.
    """
    def __init__(self):
        pass

    def setup(self, ws, data_provider, *args, **kwargs):
        ws.jacobianSetFuncTransformation(transformation_func = "log")

    def __call__(self, x):
        return np.log(x)

    def invert(self, y):
        return np.exp(1) ** y

class Atanh(Transformation):

    def __init__(self, z_min = 0.0, z_max = 1.0):
        Transformation.__init__(self)
        ArtsObject.__init__(self)

        self.z_min = z_min
        self.z_max = z_max

    @arts_property("Numeric")
    def z_min(self):
        return 0.0

    @arts_property("Numeric")
    def z_max(self):
        return 1.2

    def setup(self, ws, data_provider, *args, **kwargs):
        ws.jacobianSetFuncTransformation(transformation_func = "atanh",
                                         z_min = self.z_min,
                                         z_max = self.z_max)

    def __call__(self, x):
        x = np.minimum(x, 0.99 * self.z_max)
        x = np.maximum(x, self.z_min)
        return np.arctanh(2.0 * (x - self.z_min) / (self.z_max - self.z_min) - 1)

    def invert(self, y):
        return (np.tanh(y) + 1) * 0.5 * (self.z_max - self.z_min) + self.z_min

class Identity(Transformation):
    """
    The identity transformation $f(x) = x$.
    """
    def __init__(self):
        pass

    def setup(self, ws, data_provider, *args, **kwargs):
        pass

    def __call__(self, x):
        return x

    def invert(self, y):
        return y

class Composition(Transformation):
    """
    Composition of multiple transformations.

    The forward transformation is applied left to right
    as provided to the constructor.

    Arguments:
        *args: Sequence of transformations.
    """
    def __init__(self, *args):
        if not all([isinstance(a, Transformation) for a in args]):
            raise Exception("All provided transformation must implement the "
                            "abstract base class.")
        self.transformations = args


        if any([hasattr(t, "initialize") for t in self.transformations]):
            self.__dict__["initialize"] = self._initialize

    def _initialize(self, data_provider, *args, **kwargs):
        for t in self.transformations:
            if hasattr(t, "initialize"):
                t.initialize(data_provider, *args, **kwargs)

    def setup(self, ws, data_provider, *args, **kwargs):
        for t in self.transformations:
            t.setup(ws, data_provider, *args, **kwargs)

    def __call__(self, x):
        for t in self.transformations:
            x = t(x)
        return x

    def invert(self, y):
        for t in self.transformations[::-1]:
            y = t.invert(y)
        return y

################################################################################
# JacobianCalculation
################################################################################

class JacobianCalculation:
    """
    The :class:`JacobianCalculation` keeps track of the quantities for which
    to compute a Jacobian and coordinates their setup.
    """
    def __init__(self):
        """
        Initialize an empty Jacobian calculation.
        """
        self.jacobian_quantities = []

    def add(self, jq):
        """
        Add a quantity to list of quantities for which the Jacobian should
        be computed.

        Arguments:

            jq(:class:`JacobianQuantity`): Add Jacobian calculation for the given
            quantity. The quantity :code:`jq` must be an instance of the
            :class:`JacobianQuantity` abstract base class.
        """
        jq.jacobian = jq.jacobian_class(jq, len(self.jacobian_quantities))
        self.jacobian_quantities += [jq]

    def setup(self, ws, data_provider, *args, **kwargs):
        """
        Setup the Jacobian calculations on the given workspace.

        Initializes the definition of the Jacobian on the given workspace
        :code:`ws` by calling the :code:`jacobianInit` WSV on the given
        workspace. Calls the setup method for all quantities for which
        a Jacobian should be computed and finalizes the definition of
        the Jacobian by calling the :code:`jacobianClose` WSM.

        Arguments:

            ws(:code:`pyarts.workspace.Workspace`): Workspace object
            on which to setup the Jacobian calculation.

            data_provider: The data provider object providing the data for
            the simulation.

        """
        if not self.jacobian_quantities:
            ws.jacobianOff()
        else:
            ws.jacobianInit()
            for jq in self.jacobian_quantities:
                jq.jacobian.setup(ws)
                jq.transformation.setup(ws, data_provider, *args, **kwargs)
            ws.jacobianClose()


################################################################################
# JacobianQuantity
################################################################################

class JacobianQuantity(metaclass = ABCMeta):
    """
    Abstract interface for quantities for which a Jacobian can be computed.

    Quantities for which a Jacobian can be computed must expose a
    :code:`jacobian_class` which holds all quantity-specific WSM calls and
    settings required to compute its Jacobian.

    After a quantity has been added to the Jacobian quantities of a simulation,
    the :code:`jacobian_class` object representing the settings and results of
    the Jacobian calculation for this specific object can be accessed through
    its :code:`jacobian` property.
    """

    def __init__(self):
        self._transformation = None
        self._jacobian = None

    @abstractproperty
    def jacobian_class(self):
        """
        Return the class object that holds the actual implementation of the
        Jacobian calculation.
        """
        pass

    @property
    def jacobian(self):
        """
        The :code:`jacobian_class` object holding the quantity-specific settings
        and actual results of the Jacobian calculations for this quantity.
        """
        return self._jacobian

    @jacobian.setter
    def jacobian(self, j):
        if not isinstance(j, self.jacobian_class):
           #not isinstance(j, JacobianBase):
            raise ValueError("The jacobian property of a JacobianQuantity"\
                             " can only be set to an instance of the objects"\
                             "own jacobian_class.")
        else:
            self._jacobian = j

    @property
    def transformation(self):
        """
        The transformation to be applied to the retrieval quantity.
        """
        return self._transformation

    @transformation.setter
    def transformation(self, t):
        if not isinstance(t, Transformation):
            raise TypeError("The transformation of a retrieval quantity must"\
                            " be of type Transformation.")
        else:
            self._transformation = t

################################################################################
# JacobianBase
################################################################################

class JacobianBase(ArtsObject, metaclass = ABCMeta):
    """
    Abstract base class for the Jacobian classes that encapsulate the
    quantity-specific calls and settings of the Jacobian. This class
    muss be inherited by the :code:`jacobian_class` of each
    :code:`JacobianQuantity` object.
    """
    @arts_property("Vector", shape = (dim.Joker,))
    def p_grid(self):
        return np.zeros(0)

    @arts_property("Vector", shape = (dim.Joker,))
    def lat_grid(self):
        return np.zeros(0)

    @arts_property("Vector", shape = (dim.Joker,))
    def lon_grid(self):
        return np.zeros(0)

    def __init__(self, quantity, index):

        super().__init__()
        self.quantity = quantity
        self.index    = index

        if self.quantity.transformation is None:
            self.quantity.transformation = Identity()

    @property
    def name(self):
        return self.quantity.name

    @abstractmethod
    def setup(self, ws):
        """
        This method should call the :code:`jacobianAdd...` method corresponding to
        the quantity on the given workspace :code:`ws`.
        """
        pass

    def get_grids(self, ws):

        if self.p_grid.size == 0:
            g1 = ws.p_grid
        else:
            g1 = self.p_grid

        if self.lat_grid.size == 0:
            g2 = ws.lat_grid
        else:
            g2 = self.lat_grid

        if self.lon_grid.size == 0:
            g3 = ws.lon_grid
        else:
            g3 = self.lon_grid

        return {"g1" : g1, "g2" : g2, "g3" : g3}

    def interpolate_to_grids(self, x, grids):
        retrieval_grids = [self.p_grid,
                           self.lat_grid,
                           self.lon_grid]
        used_grids = []
        for i, g in enumerate(grids):
            if retrieval_grids[i].size == 0:
                used_grids += [g]
            else:
                used_grids += [retrieval_grids[i]]
        retrieval_grids = used_grids

        retrieval_grids_shape = [g.size for g in retrieval_grids]
        x = np.reshape(x, retrieval_grids_shape)

        x = x[::-1]
        retrieval_grids[0] = retrieval_grids[0][::-1]
        grids[0] = grids[0][::-1]

        interp = sp.interpolate.RegularGridInterpolator(retrieval_grids, x,
                                                        method = "linear",
                                                        bounds_error = False,
                                                        fill_value = np.nan)

        mesh_grids = np.meshgrid(grids)
        if len(mesh_grids) > 1:
            xi = np.transpose(np.stack(mesh_grids), axes = (0, -1))
        else:
            xi = mesh_grids[0].reshape(-1, 1)

        y  = interp(xi)

        inds = np.array(np.isnan(y))
        interp = sp.interpolate.RegularGridInterpolator(retrieval_grids, x,
                                                        method = "nearest",
                                                        bounds_error = False,
                                                        fill_value = None)
        y[inds] = interp(xi)[inds]

        grids_shape = [g.size for g in grids]
        y = np.reshape(y, grids_shape)[::-1]

        return y

    def get_data(self, ws, data_provider, *args, **kwargs):
        self.get_data_arts_properties(ws, data_provider, *args, **kwargs)
