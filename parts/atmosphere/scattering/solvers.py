import numpy as np
from abc import ABCMeta, abstractproperty, abstractmethod
from typhon.arts.workspace.methods import workspace_methods
from typhon.arts.workspace.variables import workspace_variables

wsv = workspace_variables
wsm = workspace_methods

class ScatteringSolver(metaclass = ABCMeta):
    def __init__(self):
        pass

    @abstractproperty
    def solver_call(self):
        pass

    @abstractproperty
    def solver_args(self):
        pass

    @abstractproperty
    def solver_kwargs(self):
        pass

class RT4(ScatteringSolver):
    def __init__(self,
                 nstreams = 8,
                 pfct_method = "median",
                 quad_type = "D",
                 add_straight_angles = 1,
                 pfct_aa_grid_size = 38,
                 auto_inc_nstreams = 8,
                 robust = 1):
        self._nstreams = nstreams
        self._pfct_method = pfct_method
        self._quad_type = quad_type
        self._add_straight_angles = add_straight_angles
        self._pfct_aa_grid_size = pfct_aa_grid_size
        self._auto_inc_nstreams = auto_inc_nstreams
        self._robust = robust

    #
    # Properties
    #

    @property
    def nstreams(self):
        return self._nstreams

    @property
    def pfct_method(self):
        return self._pfct_method

    @property
    def quad_type(self):
        return self._quad_type

    @property
    def add_straight_angles(self):
        return self._add_straight_angles

    @property
    def pfct_aa_grid_size(self):
        return self._pfct_aa_grid_size

    @property
    def auto_inc_nstreams(self):
        return self._auto_inc_nstreams

    @property
    def robust(self):
        return self._robust

    @property
    def solver_call(self):
        return wsm["RT4Calc"]

    @property
    def solver_args(self):
        return []

    @property
    def solver_kwargs(self):
       return {"nstreams" : self.nstreams,
               "pfct_method" : self.pfct_method,
               "quad_type" : self.quad_type,
               "add_straight_angles" : self.add_straight_angles,
               "pfct_aa_grid_size" : self.pfct_aa_grid_size,
               "auto_inc_nstreams" : self.auto_inc_nstreams,
               "robust" : self.robust}

