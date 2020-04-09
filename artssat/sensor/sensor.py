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
               from 'pyarts.workspace.methods'
    wsv(dict): Alias for the :code:`workspace_variables` dictionary
               from :code:`pyarts.workspace.variables`
"""
from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np

from artssat.arts_object import ArtsObject, arts_property
from artssat.arts_object import Dimension as dim
from pyarts.types import SingleScatteringData
from pyarts.workspace import Workspace, arts_agenda
from pyarts.workspace.methods import workspace_methods
from pyarts.workspace.variables import WorkspaceVariable, \
                                            workspace_variables

wsv = workspace_variables
wsm = workspace_methods

################################################################################
### Abstract sensor class
################################################################################

class Sensor(ArtsObject):
    """
    Defines an interface and implementes common functionality for
    classes representing ARTS sensors.

    Declares ARTS properties common to all types of sensors handled
    by ARTS.

    Arts Attributes:

        properties: The ARTS properties that hold the data common to
            all sensors in ARTS.
    """

    private_wsvs = ["f_grid", "scat_data", "scat_data_checked",
                    "iy_unit", "iy_aux_vars", "sensor_los", "sensor_pos",
                    "sensor_time", "mblock_dlos_grid", "sensor_response",
                    "antenna_dim", "sensor_norm", "sensor_response",
                    "sensor_response_f", "sensor_response_f_grid",
                    "sensor_response_dlos", "sensor_response_dlos_grid",
                    "sensor_response_pol", "sensor_response_pol_grid",
                    "iy_main_agenda"]


    ############################################################################
    # ARTS properties
    ############################################################################

    @arts_property("Vector",
                   shape = (dim.Joker,),
                   wsv = wsv["f_grid"])
    def f_grid(self):
        """
        The frequency grid of the sensor.
        """
        return None

    @arts_property("String",
                   wsv = wsv["iy_unit"])
    def iy_unit(self):
        """
        The unit which to use for the measurement vector.
        """
        return "1"

    @arts_property("ArrayOfString",
                   shape = (dim.Joker,),
                   wsv = wsv["iy_aux_vars"])
    def iy_aux_vars(self):
        """
        Which auxilliary variables to compute dunring the simulation.
        """
        return []

    @arts_property("Integer", wsv = wsv["stokes_dim"])
    def stokes_dimension(self):
        """
        Which stokes dimensions to use for the simulations.
        """
        return 1

    @arts_property("Matrix",
                   shape = (dim.Joker, dim.Atm),
                   wsv = wsv["sensor_pos"])
    def sensor_position(self):
        """
        For which sensor positions ot perform the simulations.
        """
        return []

    @arts_property("Matrix",
                   shape = (dim.Joker, dim.Atm),
                   wsv = wsv["transmitter_pos"])
    def transmitter_pos(self):
        """
        Positions of the transmitter for radio link calculations.
        """
        return []

    @arts_property("Matrix",
                   shape = (dim.Obs, dim.Los),
                   wsv = wsv["sensor_los"])
    def sensor_line_of_sight(self):
        """
        Line of sight for each sensor position.
        """
        return []

    @arts_property("Matrix",
                   shape = (dim.Obs, dim.Los),
                   wsv = wsv["mblock_dlos_grid"])
    def sensor_line_of_sight_offsets(self):
        """
        Line of sight offsets for each measurment block.
        """
        return np.zeros((1, self.sensor_line_of_sight.shape[1]))

    @arts_property("Sparse",
                   shape = (dim.Joker, dim.Joker),
                   wsv = wsv["sensor_response"])
    def sensor_response(self):
        """
        The sensor response matrix.
        """
        return []

    @arts_property("Vector",
                   shape = (dim.Joker,),
                   wsv = wsv["sensor_response_f"])
    def sensor_response_f(self):
        """
        The frequencies associated with the rows of the sensor response matrix.
        """
        return []

    @arts_property("ArrayOfIndex",
                   shape = (dim.Joker,),
                   wsv = wsv["sensor_response_pol"])
    def sensor_response_pol(self):
        """
        The polarization states associated with the rows of the sensor reponse
        matrix.
        """
        return []

    @arts_property("Vector",
                   shape = (dim.Joker, dim.Joker),
                   wsv = wsv["sensor_response_dlos"])
    def sensor_response_dlos(self):
        """
        The line-of-sight offsets associates with the rows of the sensor
        response matrix.
        """
        return []

    @arts_property("Vector",
                   shape = (dim.Joker,),
                   wsv = wsv["sensor_response_f_grid"])
    def sensor_response_f_grid(self):
        """
        The frequency grid associated with sensor response.
        """
        return []

    @arts_property("ArrayOfInteger",
                   shape = (dim.Joker,),
                   wsv = wsv["sensor_response_pol_grid"])
    def sensor_response_pol_grid(self):
        """
        The polarization states associated with sensor response.
        """
        return []

    @arts_property("Vector",
                   shape = (dim.Joker,),
                   wsv = wsv["sensor_response_dlos_grid"])
    def sensor_response_dlos_grid(self):
        """
        The LOS offsets associated with sensor response.
        """
        return []

    @arts_property("Vector",
                   shape = (dim.Joker,),
                   wsv = wsv["sensor_time"])
    def sensor_time(self):
        """
        The times associated with each measurment block.
        """
        return np.array(np.zeros(self.sensor_position.shape[0]))

    @arts_property("Index",
                   wsv = wsv["sensor_norm"])
    def sensor_norm(self):
        """
        Flag whether or not to normalize the sensor response.
        """
        return 1
    @arts_property("Index",
                   wsv = wsv["antenna_dim"])
    def antenna_dim(self):
        """
        The dimensionality of the antenna pattern.
        """
        return 1

    ############################################################################
    # Class methods
    ############################################################################

    def __init__(self, name, f_grid = None, stokes_dimension = 1):
        """
        Create a sensor with given frequency grid :code:`f_grid` and
        stokes dimension :code: `stokes_dimension`
        """
        super().__init__()
        self._wsvs = {}

        if not f_grid is None:
            self.f_grid = f_grid

        self.name = name
        self.stokes_dimension = stokes_dimension

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

    @stokes_dimension.setter
    def stokes_dimension_setter(self, n):
        """
        Specialized setter that overwrites the default from the
        :code:`ArtsObject` base class. Makes sure that the stokes
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

    @abstractproperty
    def y_vector_length(self):
        pass

    @property
    def views(self):
        return self.sensor_position.shape[0]

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

            ws(pyarts.workspace.Workspace): The workspace on which
                to perform the setup of the sensor.
        """
        self._create_private_wsvs(ws, type(self).private_wsvs)
        wsvs = self._wsvs

        self.setup_arts_properties(ws)

        #
        # Scat data
        #
        if ws.scat_data_raw.initialized:
            ws.scat_dataCalc(scat_data_raw=ws.scat_data_raw,
                             f_grid=wsvs["f_grid"],
                             interp_order=1)
            ws.Copy(wsvs["scat_data"], ws.scat_data)

        kwargs = self.get_wsm_kwargs(wsm["scat_data_checkedCalc"])
        ws.scat_data_checkedCalc(**kwargs, check_level = "sane")
        wsvs["scat_data_checked"].value = ws.scat_data_checked.value

        #
        # Need to add agendas in the end so that input arguments
        # can be replaced by private sensor variables.
        #
        wsvs["iy_main_agenda"].value = self.make_iy_main_agenda(scattering)

    def get_data(self, ws, data_provider, *args, **kwargs):
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
        self.get_data_arts_properties(ws, data_provider, *args, **kwargs)
        if isinstance(self.sensor_response, list) and self.sensor_response == []:
            self.call_wsm(ws, wsm["sensor_responseInit"])
        self.call_wsm(ws, wsm["sensor_checkedCalc"])



################################################################################
### Active sensor class
################################################################################

class ActiveSensor(Sensor):
    """
    Specialization of the abstract :code:`Sensor` class that implements
    active sensors (Radar).

    """
    ws = Workspace()
    extinction_scaling = ws.create_variable("Numeric", "extinction_scaling")
    private_wsvs = Sensor.private_wsvs + ["range_bins",
                                          "instrument_pol_array",
                                          "instrument_pol",
                                          "iy_transmitter_agenda",
                                          "extinction_scaling"]

    ############################################################################
    # ARTS properties
    ############################################################################

    @arts_property("Numeric", wsv = "extinction_scaling")
    def extinction_scaling(self):
        return 1.0

    @arts_property("Numeric")
    def y_min(self):
            return -35.0

    @arts_property("Vector", shape = (dim.Joker,), wsv = wsv["range_bins"])
    def range_bins(self):
        return []

    @arts_property("ArrayOfIndex",
                   wsv = wsv["instrument_pol"])
    def instrument_pol(self):
        return [1]

    @arts_property("ArrayOfArrayOfIndex",
                   wsv = wsv["instrument_pol_array"])
    def instrument_pol_array(self):
        return [[1]]

    @property
    def y_vector_length(self):
        return (self.range_bins.size - 1) * self.f_grid.size * self.stokes_dimension

    def __init__(self, name, f_grid, stokes_dimension, range_bins = None):
        super().__init__(name, f_grid, stokes_dimension = stokes_dimension)
        self.iy_unit    = "dBZe"

        if not range_bins is None:
            self.range_bins = range_bins

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
        kwargs = self.get_wsm_kwargs(wsm["iy_transmitterSinglePol"])
        @arts_agenda
        def iy_transmitter_agenda(ws):
            ws.Ignore(ws.rtp_pos)
            ws.Ignore(ws.rtp_los)
            ws.Ignore(ws.f_grid)
            ws.iy_transmitterSinglePol(**kwargs)

        return iy_transmitter_agenda

    def make_iy_main_agenda(self, scattering = False):
        """
        The :code: `iy_main_agenda` for active sensor. Currently uses
        the single scattering radar module, but might be extended
        at some point.
        """

        kwargs = self.get_wsm_kwargs(wsm["iyActiveSingleScat2"])
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
            ws.iyActiveSingleScat(**kwargs,
                                  pext_scaling = self._wsvs["extinction_scaling"],
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
            self._iy_unit.value = u
            self._iy_unit.fixed = True

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
            self._iy_aux_vars.value = v
            self._iy_aux_vars.fixed = False

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
            ws.IndexSet(ws.stokes_dim, self.stokes_dimension)
            #ws.Copy(ws.iy_transmitter_agenda,
            #        self._wsvs["iy_transmitter_agenda"])
            #ws.Copy(ws.iy_main_agenda,
            #        self._wsvs["_iy_main_agenda"])
            #ws.Copy(ws.instrument_pol, self._wsvs["instrument_pol"])
            #ws.IndexSet(self._wsvs["_stokes_dim"], self.stokes_dimension)
            #ws.IndexSet(ws.stokes_dim, self.stokes_dimension)

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

        kwargs = self.get_wsm_kwargs(wsm["yActive"])

        if self.y_min:
            y_min = self.y_min
        else:
            y_min = - np.inf

        def y_calc(ws):
            ws.yActive(**kwargs, dbze_min = y_min)

        return y_calc

    #
    # Setup
    #

    def setup(self, ws, scattering = True):
        super().setup(ws, scattering)
        self._wsvs["iy_transmitter_agenda"].value = self.iy_transmitter_agenda

class PassiveSensor(Sensor):
    """
    Specialization of the abstract Sensor class for passive sensors.
    """

    @arts_property("Index")
    def t_interp_order(self):
        return 0

    @property
    def y_vector_length(self):
        if hasattr(self.sensor_response, "shape"):
            return self.sensor_response.shape[0]
        else:
            return self.f_grid.size * self.stokes_dimension

    def __init__(self, name, f_grid, stokes_dimension = 1):
        """
        Paramters:
            f_grid(numpy.ndarray) The frequency grid of the sensor.
            stokes_dimension(int) The stokes dimensions to use for simulating
                the sensor measurement. Must be one of [1, 2, 4].
        """
        super().__init__(name, f_grid, stokes_dimension)
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
        def iy_main_agenda_scattering(ws):
            ws.Ignore(ws.iy_id)
            ws.Ignore(ws.nlte_field)
            ws.Ignore(ws.rte_pos2)
            ws.Ignore(ws.iy_unit)
            ws.Ignore(ws.iy_aux_vars)
            ws.FlagOff(ws.cloudbox_on)
            ws.ppathCalc()
            ws.FlagOn(ws.cloudbox_on)
            ws.iyHybrid2(**kwargs, t_interp_order = self.t_interp_order)

        def iy_main_agenda_no_scattering(ws):
            ws.Ignore(ws.iy_id)
            ws.Ignore(ws.nlte_field)
            ws.Ignore(ws.rte_pos2)
            ws.Ignore(ws.iy_unit)
            ws.Ignore(ws.iy_aux_vars)
            ws.ppathCalc()
            ws.iyEmissionStandard(**kwargs)

        if scattering:
            agenda = iy_main_agenda_scattering
            kwargs = self.get_wsm_kwargs(wsm["iyHybrid2"])
        else:
            agenda = iy_main_agenda_no_scattering
            kwargs = self.get_wsm_kwargs(wsm["iyEmissionStandard"])

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
            kwargs = self.get_wsm_kwargs(wsm["yCalcAppend"])
        else:
            kwargs = self.get_wsm_kwargs(wsm["yCalc"])

        def y_calc_append(ws):
            ws.yCalcAppend(**kwargs,
                           jacobian_quantities_copy = ws.jacobian_quantities)

        def y_calc(ws):
            ws.yCalc(**kwargs)

        if append:
            f = y_calc_append
        else:
            f = y_calc

        return f


################################################################################
# Ice cloud imager (ICI).
################################################################################

class ICI(PassiveSensor):
    """
    The Ice Cloud Imager (ICI) sensor.

    Attributes:

        channels(:code:`list`): List of channels that are available
            from ICI

        nedt(:code:`list`): Noise equivalent temperature differences for the
            channels in :code:`channels`.
    """
    channels = np.array([1.749100000000000e+11,
                         1.799100000000000e+11,
                         1.813100000000000e+11,
                         2.407000000000000e+11,
                         3.156500000000000e+11,
                         3.216500000000000e+11,
                         3.236500000000000e+11,
                         4.408000000000000e+11,
                         4.450000000000000e+11,
                         4.466000000000000e+11,
                         6.598000000000000e+11])

    nedt = np.array([0.8, 0.8, 0.8,       # 183 GHz
                     0.7 * np.sqrt(0.5),  # 243 GHz
                     1.2, 1.3, 1.5,       # 325 GHz
                     1.4, 1.6, 2.0,       # 448 GHz
                     1.6 * np.sqrt(0.5)]) # 664 GHz

    def __init__(self,
                 name = "ici",
                 channel_indices = None,
                 stokes_dimension = 1,
                 lines_of_sight = None,
                 positions = None):
        """
        This creates an instance of the ICI sensor to be used within a
        :code:`artssat` simulation.

        Arguments:

            name(:code:`str`): The name of the sensor used within the artssat
                simulation.

            channel_indices(:code:`list`): List of channel indices to be used
                in the simulation/retrieval.

            stokes_dimension(:code:`int`): The stokes dimension to use for
                the retrievals.
        """
        if channel_indices is None:
            channels  = ICI.channels
            self.nedt = ICI.nedt
        else:
            channels  = ICI.channels[channel_indices]
            self.nedt = self.nedt[channel_indices]
        super().__init__(name, channels, stokes_dimension = stokes_dimension)

        if not (lines_of_sight is None):
            if not (positions is None):
                self.sensor_line_of_sight = lines_of_sight
                self.sensor_position = positions
            else:
                self.sensor_position = np.array([[600e3]])
                self.sensor_line_of_sight = np.array([[135.0]])
        else:
            self.sensor_position = np.array([[600e3]])
            self.sensor_line_of_sight = np.array([[135.0]])

################################################################################
# Microwave imager (MWI).
################################################################################

class MWI(PassiveSensor):
    """
    The Microwave Imager (MWI) sensor.

    Attributes:

        channels(:code:`list`): The list of the channels available from the
            MWI sensor.

        nedt(:code:`list`): The noise equivalent temperature differences for
            the channels in :code:`channels`.
    """
    channels = np.array([18.7e9,
                         23.8e9,
                         31.4e9,
                         50.3e9,
                         52.6e9,
                         53.24e9,
                         53.75e9,
                         89.0e9,
                         115.5503e9,
                         116.6503e9,
                         117.3503e9,
                         117.5503e9,
                         164.75e9,
                         176.31e9,
                         177.21e9,
                         178.41e9,
                         179.91e9,
                         182.01e9])

    nedt = np.array([0.8 * np.sqrt(0.5), #18 GHz
                     0.7 * np.sqrt(0.5), #24 GHz
                     0.9 * np.sqrt(0.5), #31 GHz
                     1.1 * np.sqrt(0.5), #50 GHz
                     1.1 * np.sqrt(0.5),
                     1.1 * np.sqrt(0.5),
                     1.1 * np.sqrt(0.5),
                     1.1 * np.sqrt(0.5), #89 GHz
                     1.3, #118 GHz
                     1.3,
                     1.3,
                     1.3,
                     1.2, #165 GHz
                     1.3, #183 GHz
                     1.2,
                     1.2,
                     1.2,
                     1.3])

    def __init__(self,
                 name = "mwi",
                 channel_indices = None,
                 stokes_dimension = 1):
        """
        Create an MWI instance to be used within a :code:`artssat` simulation.

        Arguments:

            name(:code:`str`): The name of the sensor to be used within the
                artssat simulation.

            channel_indices(:code:`list`): List of channel indices to be used
                for the simulation.

            stokes_dimension(:code:`int`): The Stokes dimension to be used for
                the simulation.
        """
        if channel_indices is None:
            channels  = MWI.channels
            self.nedt = MWI.nedt
        else:
            channels  = MWI.channels[channel_indices]
            self.nedt = MWI.nedt[channel_indices]

        self.channels = channels
        super().__init__(name, channels, stokes_dimension = stokes_dimension)

class CloudSat(ActiveSensor):
    channels = np.array([94.0e9])

    def __init__(self,
                 name = "cloud_sat",
                 range_bins = np.arange(500.0, 20e3, 500.0),
                 stokes_dimension = 2,
                 lines_of_sight = None,
                 positions = None):
        super().__init__(name,
                         f_grid = np.array([94e9]),
                         stokes_dimension = stokes_dimension,
                         range_bins = range_bins)
        self.instrument_pol       = [1]
        self.instrument_pol_array = [[1]]
        self.extinction_scaling   = 1.0
        self.y_min = -30.0

        if not (lines_of_sight is None):
            if not (positions is None):
                self.sensor_line_of_sight = lines_of_sight
                self.sensor_position = positions
            else:
                self.sensor_position = np.array([[600e3]])
                self.sensor_line_of_sight = np.array([[180.0]])
        else:
            self.sensor_position = np.array([[600e3]])
            self.sensor_line_of_sight = np.array([[180.0]])

ici = ICI()
