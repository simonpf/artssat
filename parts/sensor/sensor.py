""" ARTS Sensors

Class hierarchy that represents the different types of sensors
in ARTS. The purpose of this hierarchy is to encapsulate all
required settings and configurations related to the sensor
in an ARTS simulation.

An ARTS simulation contains one or more sensors. The interaction
between ARTS simulation and sensor object is structured as follows:

1. Setup: During the setup phase the ARTS simulation calls the
   setup method of each sensor object. During this call the sensor
   is expected to setup all necessary workspace variables on
   the provided workspace and perform all necessary preparatory
   calculations, such as calculating absorption and scattering
   data.

2. Run: During the run phase, the simulation calls the get_data(...)
   method of the sensor during which the sensor is supposed to
   request all necessary data, that hasn't been set in advance.

In addition to that, the simulation needs to know which WSMs
to call to compute the measurement vector. To this end, each
sensor needs to provide the following factory methods:

1. make_preparation_function: This function should generate
   a function that runs all necessary preparations on a
   provided workspace. Currently, this requires setting
   only the stokes dimension (because of an inconsistency in ARTS).

2. make_y_calc_function: This function should generate a
   functions that runs the necessary workspace methods
   to simulate the measurements on a given workspace.

Both of these factory methods must produce functions that
can be converted into an ARTS agenda, so that they can be
used also within the inversion iterate agenda for retrievals.

In general, sensors should store important within their own,
anonymous workspace variables, which should then just be
used to replace the input WSV in all sensor-related WSM
calls. Due to the inconsistency metioned abover, however, this
principle does not work for the stokes dimension as it is
not consistently passed to the surface agenda.

Attributes:

    wsm(dict): Alias for the :code:'workspace_methods' dictionary
               from 'typhon.arts.workspace.methods'
    wsv(dict): Alias for the :code:`workspace_variables` dictionary
               from :code:`typhon.arts.workspace.variables`
"""
from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np

from parts import dimensions as dim

from parts.arts_object import ArtsObject
from typhon.arts.types import SingleScatteringData
from typhon.arts.workspace import Workspace, arts_agenda
from typhon.arts.workspace.methods import workspace_methods
from typhon.arts.workspace.variables import WorkspaceVariable, \
                                            workspace_variables

wsv = workspace_variables
wsm = workspace_methods

################################################################################
### Abstract sensor class
################################################################################

class Sensor(metaclass = ArtsObject):
    """
    Defines an interface and implementes common functionality for
    classes representing ARTS sensors.

    Declares ARTS properties common to all types of sensors handled
    by ARTS.

    Arts Attributes:

        properties: The ARTS properties that hold the data common to
            all sensors in ARTS.
    """

    properties = [("f_grid", (dim.joker), np.ndarray),
                  ("stokes_dimension", (), int),
                  ("iy_unit", (), str),
                  ("iy_aux_vars", (dim.joker), list),
                  ("sensor_position", (dim.joker, dim.atm), np.ndarray),
                  ("sensor_line_of_sight", (dim.joker, dim.los), np.ndarray),
                  ("sensor_line_of_sight_offsets", (dim.joker, dim.los),
                   np.ndarray),
                  ("sensor_response", (dim.joker, dim.joker), np.ndarray),
                  ("sensor_response_f", (dim.joker,), np.ndarray),
                  ("sensor_response_pol", (dim.joker,), list),
                  ("sensor_response_dlos", (dim.joker, dim.joker),
                   np.ndarray),
                  ("sensor_response_f_grid", (dim.joker,), np.ndarray),
                  ("sensor_response_pol_grid", (dim.joker,), list),
                  ("sensor_response_dlos_grid", (dim.joker, dim.joker),
                   np.ndarray),
                  ("sensor_norm", (1,), int),
                  ("antenna_dim", (1,), int)]

    def __init__(self, f_grid = None, stokes_dimension = 1):
        """
        Create a sensor with given frequency grid :code:`f_grid` and
        stokes dimension :code: `stokes_dimension`
        """

        self._wsvs = {}

        if not f_grid is None:
            self.f_grid = f_grid

        self.stokes_dimension = stokes_dimension
        self.sensor_norm = 1
        self.antenna_dim = 1

    def get_wsm_args(self, wsm):
        """
        Generate a list of arguments to the given ARTS workspace method
        :code:`wsm` for which the sensor related input parameters are
        replace by the ones of this sensor. This is done by checking
        whether the input argument name is in the sensors :code:`_wsv`
        dictionary and if so replacing the argument.

        Parameters:

           wsm(typhon.arts.workspace.methods.Workspacemethod): The ARTS
               workspace method object for which to generate the input
               argument list.

        Returns:

            The list of input arguments with sensor specific input arguments
            replaced by the corresponding WSVs of the sensor.

        """
        args = []
        for i in wsm.ins:
            name = WorkspaceVariable.get_variable_name(i)
            if name in self._wsvs:
                args += [self._wsvs[name]]
            else:
                args += [wsv[name]]
        return args

    #
    # Abstract properties and methods
    #

    @abstractmethod
    def make_iy_main_agenda(self, scattering = False):
        """
        Property that takes the role of a factory function that creates
        the :code:`iy_main_agenda` for the given sensor. This will
        depend on the sensor type and its specific settings, so
        must be implemented by the inhereting classes.
        """
        pass

    @abstractmethod
    def make_preparation_function(self, append = False):
        """
        Factory method for a preparation function :code:`f(ws)` which
        sets all workspace variables that are required before simulating
        a measurement on a given workspace :code:`ws`.

        The separation into a preparation function and a :code:`y_calc`
        function is currently necessary as the scattering solver can
        be chosen independently from the sensor, so the simulation object
        must be able to run the scattering solver between the sensor
        preparations and running :code:`y_calc`.

        Parameters:
            append(bool): Whether the call should use :code:`yCalcAppend`
                or not. Active sensors will probably have to throw an
                exception here.
        """
        pass

    @abstractmethod
    def make_y_calc_function(self,
                             append = False,
                             scattering = False):
        """
        Factory method that should create a function :code:`f` that runs
        the actual radiative transfer on a provided workspace.

        Parameters:
            append(bool): Whether the call should use :code:`yCalcAppend`
                or not. Active sensors will probably have to throw an
                exception here.
        """
        pass

    #
    # Special setters
    #

    def stokes_dimension_setter(self, n):
        """
        Specialized setter that overwrites the default from the
        :code:`ArtsObject` meta class. Makes sure that the stokes
        dimension is 1,2 or 4.

        Parameters:

            n(int): The new stokes dimension

        Raises:

            Exception if the new value for the stokes dimension is
            not 1, 2 or 4.
        """
        if not (n in [1, 2, 4]):
            raise Exception("Stokes dimension must be 1, 2 or 4.")
        self._stokes_dimension.fixed = True
        self._stokes_dimension.value = n

    #
    # General sensor setup
    #

    def setup(self, ws, scattering = True):
        """
        General setup for an ARTS sensor.

        This method performs the following steps to setup the sensor:

        - Copy the `iy_main_agenda` of the sensor into a private workspace
          variable in the workspace
        - Copy the `f_grid` into a private workspace variable
        - Compute and check `scat_data` and copy results into private
          workspace variables.
        - Copy `iy_aux_vars` and `iy_unit` to the workspace and store
          in prive workspace variables.

        Paremters:

            ws(typhon.arts.workspace.Workspace): The workspace on which
                to perform the setup of the sensor.
        """
        wsvs = self._wsvs

        wsvs["f_grid"] = ws.add_variable(self.f_grid)

        # Scat data
        name = "scat_data_" + str(id(self))
        ws.ArrayOfArrayOfSingleScatteringDataCreate(name)
        wsvs["scat_data"] = ws.__getattr__(name)
        ws.scat_dataCalc(ws.scat_data_raw, wsvs["f_grid"], interp_order = 1)
        ws.Copy(wsvs["scat_data"], ws.scat_data)

        args = self.get_wsm_args(wsm["scat_data_checkedCalc"])
        ws.scat_data_checkedCalc(*args)
        wsvs["scat_data_checked"] = ws.add_variable(1)

        # iy_aux_vars
        name = "iy_aux_vars_" + str(id(self))
        ws.ArrayOfStringCreate(name)
        wsvs["iy_aux_vars"] = ws.__getattr__(name)
        if self.iy_aux_vars is None or len(self.iy_aux_vars) == 0:
            ws.Touch(wsvs["iy_aux_vars"])
        else:
            ws.ArrayOfStringSet(wsvs["iy_aux_vars"],
                                self.iy_aux_vars)

        wsvs["iy_unit"] = ws.add_variable(self.iy_unit)

        # Stokes dimension

        wsvs["stokes_dim"] = ws.add_variable(self.stokes_dimension)

        # los, pos, offsets and response
        wsvs["sensor_los"] = ws.add_variable(np.zeros((0, 0)))
        wsvs["sensor_pos"] = ws.add_variable(np.zeros((0, 0)))
        wsvs["mblock_dlos_grid"] = ws.add_variable(np.zeros((0, 0)))
        wsvs["sensor_response"] = ws.add_variable(np.zeros((0, 0)))

        # sensor response
        wsvs["antenna_dim"] = ws.add_variable(self.antenna_dim)
        wsvs["sensor_norm"] = ws.add_variable(self.sensor_norm)

        name = "sensor_response_" + str(id(self))
        ws.SparseCreate(name)
        wsvs["sensor_response"] = getattr(ws, name)

        wsvs["sensor_response_f"] = ws.add_variable(np.zeros((0,)))
        wsvs["sensor_response_f_grid"] = ws.add_variable(np.zeros((0,)))
        wsvs["sensor_response_dlos"] = ws.add_variable(np.zeros((0, 0)))
        wsvs["sensor_response_dlos_grid"] = ws.add_variable(np.zeros((0, 0)))

        name = "sensor_response_pol_" + str(id(self))
        ws.ArrayOfIndexCreate(name)
        wsvs["sensor_response_pol"] = getattr(ws, name)
        name = "sensor_response_pol_grid_" + str(id(self))
        ws.ArrayOfIndexCreate(name)
        wsvs["sensor_response_pol_grid"] = getattr(ws, name)

        # Need to add agendas in the end so that input arguments
        # can be replaced by private sensor variables.
        wsvs["iy_main_agenda"] = ws.add_variable(
            self.make_iy_main_agenda(scattering))

    def get_data(self, ws, provider, *args, **kwargs):
        """
        Get required data from data provided.

        This function obtains required data from the data provider
        if it has not been fixed in advance. The data expected
        from the data provider are the following:

        - the line of sight data by calling the :code:`get_line_of_sight`
          method of the provided data provider.
        - the sensor position data by calling the :code: `get_sensor_pos`
          method of the provided data provider.
        - the pencil beam offsets by calling the :code:
          `get_line_of_sight_offsets`

        This function also checks the sensor data by calling
        :code:`sensor_checkedCalc` and stores the result in a private
        WSV.

        """

        wsvs = self._wsvs

        # TODO: Use easel.dimension.atm here
        dim = ws.atmosphere_dim.value

        # Sensor line of sight

        if self._sensor_line_of_sight.fixed:
            los = self.sensor_line_of_sight
        else:
            los = provider.get_sensor_line_of_sight(*args, **kwargs)

        if not len(np.shape(los)) == 2:
            raise Exception("Provided line of sight must be a 2D "
                            "numpy array.")

        if dim in [1, 2] and not los.shape[1] == 1:
                raise Exception("Provided line of sight must contain "
                                " only one element along the second "
                                "dimension for 1D and 2D atmospheres.")
        if dim == 3 and not los.shape[1] == 2:
                raise Exception("Provided line of sight must contain "
                                "two elements along the second dimension"
                                " for 1D and 2D atmospheres.")
        ws.MatrixSet(wsvs["sensor_los"], los)


        # Sensor position

        if self._sensor_position.fixed:
            pos = self.sensor_position
        else:
            pos = provider.get_sensor_position(*args, **kwargs)

        if not len(np.shape(los)) == 2:
            raise Exception("Provided sensor position must be a 2D numpy "
                            " array.")

        if not dim == np.shape(los)[1]:
                raise Exception("Provided sensor position must contain "
                                "{0} elements  along the second dimension"
                                " for a {0}D atmosphere.".format(dim))
        ws.MatrixSet(wsvs["sensor_pos"], pos)

        # Line of sight offsets
        if self._sensor_line_of_sight_offsets.fixed:
            dlos = self.sensor_line_of_sight_offsets
        elif hasattr(provider, "get_sensor_line_of_sight_offsets"):
            dlos = provider.get_sensor_line_of_sight_offsets(*args, **kwargs)
        else:
            dlos = np.zeros((1, 1))

        if dlos.shape != (0,0) and not dlos.shape[1] in [1, 2]:
                raise Exception("Provided line of sight offsets must contain"
                                " one or two elements along the second "
                                "dimension.")
        ws.MatrixSet(wsvs["mblock_dlos_grid"], dlos)

        args = self.get_wsm_args(wsm["sensor_responseInit"])
        ws.sensor_responseInit(*args)
        ws.Copy(wsvs["sensor_response"], ws.sensor_response)
        ws.Copy(wsvs["sensor_response_f"], ws.sensor_response_f)
        ws.Copy(wsvs["sensor_response_f_grid"], ws.sensor_response_f_grid)
        ws.Copy(wsvs["sensor_response_pol"], ws.sensor_response_pol)
        ws.Copy(wsvs["sensor_response_pol_grid"], ws.sensor_response_pol_grid)
        ws.Copy(wsvs["sensor_response_dlos"], ws.sensor_response_dlos)
        ws.Copy(wsvs["sensor_response_dlos_grid"], ws.sensor_response_dlos_grid)

        ws.sensor_checkedCalc(*self.get_wsm_args(wsm["sensor_checkedCalc"]))
        # Check sensor data
        if not "sensor_checked" in wsvs.keys():
            wsvs["sensor_checkedCalc"] = ws.add_variable(1)
        else:
            ws.IndexSet(wsvs["sensor_checkedCalc"], 1)



################################################################################
### Active sensor class
################################################################################

class ActiveSensor(Sensor, metaclass = ArtsObject):
    """
    Specialization of the abstract :code:`Sensor` class that implements
    active sensors (Radar).

    """

    properties = {("extinction_scaling", (), np.float),
                  ("range_bins", (dim.joker,), np.ndarray),
                  ("instrument_pol_array", (dim.joker, dim.joker), list),
                  ("instrument_pol", (dim.joker,), list)}

    def __init__(self, f_grid, stokes_dimension, range_bins = None):
        super().__init__(f_grid, stokes_dimension = stokes_dimension)

        self.iy_unit = "dBZe"
        self.iy_aux_vars = []
        self.instrument_pol = [5]
        self.instrument_pol_array = [[5]]
        self.extinction_scaling = 1.0

    #
    # Agendas
    #

    @property
    def iy_transmitter_agenda(self):
        """
        The :code:`iy_transmitter_agenda` which is required for active
        sensors. Input arguments of :code:`iy_transmitter_agenda` are
        replaced by the private workspace variables of the sensor.

        Returns:

            The iy_transmitter_agenda for the active sensor.

        """
        args = self.get_wsm_args(wsm["iy_transmitterSinglePol"])
        @arts_agenda
        def iy_transmitter_agenda(ws):
            ws.Ignore(ws.rtp_pos)
            ws.Ignore(ws.rtp_los)
            ws.Ignore(ws.f_grid)
            ws.iy_transmitterSinglePol(*args)

        return iy_transmitter_agenda

    def make_iy_main_agenda(self, scattering = False):
        """
        The :code: `iy_main_agenda` for active sensor. Currently uses
        the single scattering radar module, but might be extended
        at some point.
        """

        @arts_agenda
        def iy_main_agenda(ws):
            ws.Ignore(ws.iy_id)
            ws.Ignore(ws.nlte_field)
            ws.Ignore(ws.rte_pos2)
            ws.Ignore(ws.iy_unit)
            ws.Ignore(ws.iy_aux_vars)
            ws.FlagOff(ws.cloudbox_on)
            ws.ppathCalc()
            ws.FlagOn(ws.cloudbox_on)
            ws.iyActiveSingleScat(pext_scaling = self.extinction_scaling,
                                trans_in_jacobian = 1)

        return iy_main_agenda

    #
    # Specialized setters
    #

    def iy_unit_setter(self, u):
        if not u in ["1", "Ze", "dBZe"]:
            raise Exception("Value of iy_unit for an active sensor must"
                            " be one of ['1', 'Ze', 'dBZe']")
        else:
            self._iy_unit = u

    def iy_aux_vars_setter(self, v):
        if not type(v) == list:
            v = [v]
        if not all([u in ["Radiative background", "Backsacttering",
                         "Optical depth", "Particle extinction"] for u in v]):
            raise Exception("Value of iy_aux_vars for an active sensor must"
                            " be a list consisting of the following strings: "
                            "['RadiativeBackground', 'Backscattering', "
                            "'Optical depth', Particle extinction'].")
        else:
            self._iy_aux_vars = v

    #
    # Preparation and y_calc factories.
    #

    def make_preparation_function(self):
        """
        Return the workspace preparation function, which prepares
        a workspace for simulating the signal recorded by the sensor.

        Returns:

            The function to prepare the workspace.

        """

        def preparations(ws):
            ws.Copy(ws.iy_transmitter_agenda,
                    self._wsvs["iy_transmitter_agenda"])
            ws.Copy(ws.instrument_pol, self._wsvs["instrument_pol"])
            ws.IndexSet(ws.stokes_dim, self.stokes_dimension)

        return preparations

    def make_y_calc_function(self,
                             append = False,
                             scattering = False):
        """
        Returns y_calc function, which computes the radar signal
        on an accordingly prepared workspace. This function can
        be converted into an ARTS agenda and thus included in
        the other agendas using the INCLUDE statement.

        Returns:

            The function to compute the radar signal.
        """
        if append:
            raise Exception("ARTS doesn't support appending measurements from"
                            " active sensors.")

        args = self.get_wsm_args(wsm["yActive"])

        def y_calc(ws):
            ws.yActive(*args)

        return y_calc

    #
    # Setup
    #

    def setup(self, ws):

        wsvs = self._wsvs
        wsvs["iy_transmitter_agenda"] = ws.add_variable(
            self.iy_transmitter_agenda
        )
        wsvs["instrument_pol"] = ws.add_variable(self.instrument_pol)
        wsvs["instrument_pol_array"] = ws.add_variable(
            self.instrument_pol_array
        )
        super().setup(ws)

    def get_data(self, ws, provider, *args, **kwargs):

        if self._range_bins.fixed:
            range_bins = self.range_bins
        else:
            range_bins = provider.get_range_bins(*args, **kwargs)
        range_bins = range_bins.ravel()

        if "range_bins" in self._wsvs:
            ws.VectorSet(self._wsvs["range_bins"], range_bins)
        else:
            self._wsvs["range_bins"] = ws.add_variable(range_bins)

        super().get_data(ws, provider, *args, **kwargs)

class PassiveSensor(Sensor, metaclass = ArtsObject):
    """
    Specialization of the abstract Sensor class for passive sensors.
    """

    def __init__(self, f_grid, stokes_dimension = 1):
        """
        Paramters:
            f_grid(numpy.ndarray) The frequency grid of the sensor.
            stokes_dimension(int) The stokes dimensions to use for simulating
                the sensor measurement. Must be one of [1, 2, 4].
        """
        super().__init__(f_grid, stokes_dimension)
        self.iy_unit = "PlanckBT"

    #
    # Agendas
    #

    def make_iy_main_agenda(self, scattering = False):
        """
        Factory property that returns the :code:`iy_main_agenda` that has to be used
        for passive sensors.

        Return:

            The ARTS :code:`iy_main_agenda`
        """

        for k in self._wsvs:
            print(k)
            print(self._wsvs[k].value)

        def iy_main_agenda_scattering(ws):
            ws.Ignore(ws.iy_id)
            ws.Ignore(ws.nlte_field)
            ws.Ignore(ws.rte_pos2)
            ws.Ignore(ws.iy_unit)
            ws.Ignore(ws.iy_aux_vars)
            ws.FlagOff(ws.cloudbox_on)
            ws.ppathCalc()
            ws.FlagOn(ws.cloudbox_on)
            ws.iyHybrid(*args)

        def iy_main_agenda_no_scattering(ws):
            ws.Ignore(ws.iy_id)
            ws.Ignore(ws.nlte_field)
            ws.Ignore(ws.rte_pos2)
            ws.Ignore(ws.iy_unit)
            ws.Ignore(ws.iy_aux_vars)
            ws.ppathCalc()
            ws.iyEmissionStandard(*args)

        if scattering:
            agenda = iy_main_agenda_scattering
            args = self.get_wsm_args(wsm["iyHybrid"])
        else:
            agenda = iy_main_agenda_no_scattering
            args = self.get_wsm_args(wsm["iyEmissionStandard"])

        agenda.__name__ = "iy_main_agenda"

        return arts_agenda(agenda)

    #
    # Customized setter
    #

    def iy_unit_setter(self, u):
        """
        Custom setter for :code:`iy_unit` for passive sensors. Checks
        that the argument :code:`u` has a valid value for passive simulations.

        Parameters:
            u(str): The :code:`iy_unit` to use for passive simulations. Must
                be one of :code:`["1", "RJBT", "PlackBT", "W(m^2 m sr)",
                "W/(m^2 m-1 sr)"]`

        """
        valid = ["1", "RJBT", "PlanckBT", "W/(m^2 m sr)", "W/(m^2 m-1 sr)"]
        if not u in valid:
            raise Exception("Value of iy_unit for an active sensor must"
                            " be one of {0}".format(str(valid)))
        else:
            self._iy_unit = u

    def iy_aux_vars_setter(self, v):
        """
        Custom setter for :code:`iy_aux_vars` for passive sensors. Checks
        that the argument :code:`v` has a valid value for passive simulations.

        Parameters:
            u(str): The :code:`iy_aux_vars` to use for passive simulations. Must
                be a list containing any of :code:`["Radiative background",
                "Optical depth]`

        """
        if not type(v) == list:
            v = [v]
        if not all([u in ["Radiative background", "Optical depth"]
                    for u in v]):
            raise Exception("Value of iy_aux_vars for an active sensor must"
                            " be a list consisting of the following strings: "
                            "['RadiativeBackground', 'Optical depth'].")
        else:
            self._iy_aux_vars = v

    #
    # Preparation and y_calc factory methods
    #

    def make_preparation_function(self):
        """
        Return the workspace preparation function, which prepares
        a workspace for simulating the signal recorded by the sensor.

        Returns:

            The function to prepare the workspace.

        """

        def preparations(ws):
            ws.IndexSet(ws.stokes_dim, self.stokes_dimension)
            ws.Copy(ws.iy_main_agenda, self._wsvs["iy_main_agenda"])

        return preparations

    def make_y_calc_function(self,
                             append = False,
                             scattering = False):
        """
        Factory function that produces a function that simulates a passive
        measurement with the given sensor on an accordingly prepared workspace.

        If :code:`append` is :code:`False` the returned function will use
        the :code:`yCalc` WSV to simulate the measurement. Otherwise
        :code:`yCalcAppend` will be used.

        Parameters:

            append(bool): If :code:`True` the produced function will use
                `yCalcAppend` to calculate the measured signal.
        """

        if append:
            args = self.get_wsm_args(wsm["yCalcAppend"])
        else:
            args = self.get_wsm_args(wsm["yCalc"])

        def y_calc_append(ws):
            ws.yCalcAppend(*args,
                           jacobian_quantities_copy = ws.jacobian_quantities)

        def y_calc(ws):
            ws.yCalc(*args)

        if append:
            f = y_calc_append
        else:
            f = y_calc

        return f


class ICI(PassiveSensor):
    channels = np.array([1.749100000000000e+11,
                         1.799100000000000e+11,
                         1.813100000000000e+11,
                         1.853100000000000e+11,
                         1.867100000000000e+11,
                         1.917100000000000e+11,
                         2.407000000000000e+11,
                         2.457000000000000e+11,
                         3.156500000000000e+11,
                         3.216500000000000e+11,
                         3.236500000000000e+11,
                         3.266500000000000e+11,
                         3.286500000000000e+11,
                         3.346500000000000e+11,
                         4.408000000000000e+11,
                         4.450000000000000e+11,
                         4.466000000000000e+11,
                         4.494000000000000e+11,
                         4.510000000000000e+11,
                         4.552000000000000e+11,
                         6.598000000000000e+11,
                         6.682000000000000e+11])
    def __init__(self, channels = None):
        if channels is None:
            channels = ICI.channels
        else:
            channels = ICI.channels[channels]
        super().__init__(channels)

class ICI(PassiveSensor):
    channels = np.array([1.749100000000000e+11,
                         1.799100000000000e+11,
                         1.813100000000000e+11,
                         1.853100000000000e+11,
                         1.867100000000000e+11,
                         1.917100000000000e+11,
                         2.407000000000000e+11,
                         2.457000000000000e+11,
                         3.156500000000000e+11,
                         3.216500000000000e+11,
                         3.236500000000000e+11,
                         3.266500000000000e+11,
                         3.286500000000000e+11,
                         3.346500000000000e+11,
                         4.408000000000000e+11,
                         4.450000000000000e+11,
                         4.466000000000000e+11,
                         4.494000000000000e+11,
                         4.510000000000000e+11,
                         4.552000000000000e+11,
                         6.598000000000000e+11,
                         6.682000000000000e+11])
    def __init__(self,
                 channels = None,
                 stokes_dimension = 2):
        if channels is None:
            channels = ICI.channels
        else:
            channels = ICI.channels[channels]
        super().__init__(channels, stokes_dimension = stokes_dimension)

class CloudSat(ActiveSensor):
    channels = np.array([94.0e9])

    def __init__(self,
                 range_bins = np.arange(500.0, 20e3, 500.0),
                 stokes_dimension = 2):
        super().__init__(f_grid = np.array([94e9]),
                         stokes_dimension = stokes_dimension,
                         range_bins = range_bins)
