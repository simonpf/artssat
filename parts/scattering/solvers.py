import numpy as np
from abc import ABCMeta, abstractproperty, abstractmethod
from parts.arts_object import ArtsObject
from typhon.arts.workspace.methods import workspace_methods
from typhon.arts.workspace.variables import workspace_variables

wsv = workspace_variables
wsm = workspace_methods

class ScatteringSolver(metaclass = ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def make_solver_call(self, sensor):
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

    def make_solver_call(self, sensor):

        args = sensor.get_wsm_args(wsm["RT4Calc"])
        def run_solver(ws):
            ws.RT4Calc(*args,
                       nstreams = self._nstreams,
                       pfct_method = self._pfct_method,
                       quad_type = self._quad_type,
                       add_straight_angles = self._add_straight_angles,
                       pfct_aa_grid_size = self._pfct_aa_grid_size,
                       auto_inc_nstreams = self._auto_inc_nstreams,
                       robust = self.robust)

        return run_solver

class Disort(ScatteringSolver, metaclass = ArtsObject):

    def __init__(self,
                 nstreams = 8,
                 pfct_method = "interpolate",
                 new_optprop = 1,
                 Npfct = 181):

        self._nstreams = nstreams
        self._pfct_method = pfct_method
        self._new_optprop = new_optprop
        self._Npfct = Npfct

    def make_solver_call(self, sensor):

        args = sensor.get_wsm_args(wsm["DisortCalcWithARTSSurface"])
        args_scat_data = sensor.get_wsm_args(wsm["scat_data_checkedCalc"])
        def run_solver(ws):
            ws.Ignore(ws.atmosphere_dim)
            ws.DOAngularGridsSet(N_za_grid = 38, N_aa_grid = 1, za_grid_opt_file = "")
            ws.DisortCalcWithARTSSurface(*args, nstreams = self._nstreams,
                                         pfct_method = self._pfct_method,
                                         new_optprop = self._new_optprop,
                                         Npfct = self._Npfct)

        return run_solver

