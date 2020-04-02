from pyarts.workspace import WorkspaceVariable
from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np

class Jacobian:
    def __init__(self):
        self.method       = "analytical",
        self.perturbation = 1e-6,
        self.p_grid       = None,
        self.lat_grid     = None,
        self.lon_grid     = None

def extend_dimensions(x):
    shape = [1, 1, 1]
    if not type(x) == np.ndarray:
        x = np.array(x)
    for i, d in enumerate(x.shape):
        shape[i] = d
    return x.reshape(shape)

class AtmosphericQuantity(metaclass = ABCMeta):

    def __init__(self,
                 name,
                 dimensions,
                 jacobian = False):
        self._name = name

        self._data      = None
        self._wsv       = None
        self._wsv_index = None
        self._constant  = False

    #
    # Properties
    #

    @property
    def name(self):
        return self._name

    @property
    def dimensions(self):
        return self._dimensions

    @property
    def data(self):
        if type(self._data) == WorkspaceVariable:
            return self._data.value
        else:
            return self._data

    @data.setter
    def data(self, d):

        for i, d in enumerate(self.dimensions):
            if not d == 0 and not d == d.shape[i]:
                raise Exception("Dimensions of provided atmospheric {0}"
                                " field are inconsistent with the atmospheric"
                                " dimensions along dimension {1} ({2} vs. {3})."
                                .format(self.name, i, d, d.shape[i]))
        self._data = d
        self._constant = d

    @property
    def constant(self):
        return self._constant

    @constant.setter
    def constant(self, c):
        self._constant = c

    #
    # Methods
    #

    @abstractmethod
    def setup(self, ws, i):
        pass

    @abstractmethod
    def get_data(self, ws, provider, *args, **kwargs):
        dimensions = ws.t_field.value.shape
        f = provider.__getattribute__("get_" + self.name)
        x = f(*args, **kwargs)
        x = extend_dimensions(x)

        if not x.shape == dimensions:
            raise Exception("Shape of {0} field is inconsistent with "
                            "the dimensions of the atmosphere."
                            .format(self.name))

        self._wsv.value[self._wsv_index, :, :, :] = x
