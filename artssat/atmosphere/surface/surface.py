"""
The surface sub-module provides implementations of the different surface models
that are available in ARTS.
"""
from abc import abstractmethod, abstractproperty
import numpy as np

from artssat.arts_object import ArtsObject, arts_property, Dimension
from pyarts.workspace import Workspace, arts_agenda
from pyarts.workspace.variables import WorkspaceVariable, \
                                            workspace_variables
wsv = workspace_variables

class Surface:
    """
    Abstract base class for surfaces.

    This class defines the general interfaces for surface models in artssat.
    """

    @abstractproperty
    def required_data(self):
        pass

    @abstractmethod
    def setup(self, ws):
        """
        Setup the surface model in the given workspace.

        This method should be used to setup the surface model so that it can be
        used in a simulation.

        Args:
            ws: pyarts.workspace.Workspace object on which the simulation will
                be performed.
        """
        pass

    @abstractmethod
    def get_data(self, ws, provider, *args, **kwargs):
        """
        The method should get required data from the data provider and set required
        field in the workspace.

        Args:
            ws: pyarts.workspace.Workspace object on which the simulation will
                be performed.
            provider: The data provider for the simulations.
            *args: Arguments to be passed to the data provider.
            **kwargs: Keyword arguments to be passed to the data provider.

        """
        pass

    @abstractmethod
    def run_checks(self, workspace):
        """
        This method should run the tests that are required before the model
        can be used in a simulation.

        Args:
            workspace: The pyarts.workspace.Workspace in which the simulations are
                performed.
        """
        pass

################################################################################
# Tessem
################################################################################

class Tessem(Surface):
    """
    This class represents the " Tool to Estimate Sea‐Surface Emissivity from
    Microwaves to sub‐Millimeter waves" (TESSEM). It is a parametrization for
    the emissivity of sea surfaces that uses two neural networks to predict the
    emissivity.
    """

    def __init__(self,
                 tessem_net_h = "testdata/tessem_sav_net_V.txt",
                 tessem_net_v = "testdata/tessem_sav_net_V.txt",
                 salinity = 0.034,
                 wind_speed = 0.0):
        """
        Args:
            tessem_net_h(:code:`str`): Path to the network to use for
                h-polarization.
            tessem_net_v(:code:`str`): Path to the network to use for
                v-polarization.
            salinity (float): The salinity to assume for the sea surface.
            wind_speed (float): Wind speed if it should be set to a fixed
                value.
        """
        self.tessem_net_h = tessem_net_h
        self.tessem_net_v = tessem_net_v

        ws = Workspace()

        self._surface_temperature_field = ws.add_variable(np.zeros((1, 1)))

        self._surface_salinity       = ws.add_variable(salinity)
        self._surface_salinity_field = ws.add_variable(np.zeros((1, 1)))

        self._surface_wind_speed_field = ws.add_variable(np.zeros((1, 1)))
        self._surface_wind_speed = ws.add_variable(wind_speed)

    @property
    def required_data(self):
        return [("surface_temperature", ("n_lat", "n_lon"), False),
                ("surface_salinity", (1,), True),
                ("surface_salinity", ("n_lat", "n_lon"), True),
                ("surface_wind_speed", (1,), True),
                ("surface_wind_speed", ("n_lat", "n_lon"), True)]

    @property
    def surface_temperature_field(self):
        return self._surface_temperature_field.value

    @property
    def surface_salinity(self):
        return self._surface_salinity.value

    @property
    def wind_speed(self):
        return self._surface_wind_speed.value

    @property
    def surface_agenda(self):

        @arts_agenda
        def surface_rtprop_agenda_tessem(ws):
            ws.specular_losCalc()
            ws.InterpSurfaceFieldToPosition(out = ws.surface_skin_t,
                                            field = self._surface_temperature_field)
            ws.InterpSurfaceFieldToPosition(out = self._surface_salinity,
                                            field = self._surface_salinity_field)
            ws.InterpSurfaceFieldToPosition(out = self._surface_wind_speed,
                                            field = self._surface_wind_speed_field)
            ws.surfaceTessem(salinity = self._surface_salinity,
                             wind_speed = self._surface_wind_speed)
        return surface_rtprop_agenda_tessem

    def setup(self, ws):
        ws.TessemNNReadAscii(ws.tessem_neth, self.tessem_net_h)
        ws.TessemNNReadAscii(ws.tessem_netv, self.tessem_net_v)
        ws.Copy(ws.surface_rtprop_agenda, self.surface_agenda)

    def get_data(self, ws, provider, *args, **kwargs):

        dimensions = ws.t_field.value.shape[1:]

        x = provider.get_surface_temperature(*args, **kwargs)
        ws.MatrixSet(self._surface_temperature_field, x * np.ones(dimensions))

        if hasattr(provider, "get_surface_salinity"):
            x = provider.get_surface_salinity(*args, **kwargs)
        else:
            x = self._surface_salinity.value
        ws.MatrixSet(self._surface_salinity_field,
                     x * np.ones(dimensions))

        if hasattr(provider, "get_surface_wind_speed"):
            x = provider.get_surface_wind_speed(*args, **kwargs)
        else:
            x = self._surface_wind_speed.value
        ws.MatrixSet(self._surface_wind_speed_field,
                     x * np.ones(dimensions))

    def run_checks(self, ws):
        pass

################################################################################
# Telsem
################################################################################

class Telsem(ArtsObject):

    def __init__(self,
                 atlas_directory,
                 month=6):
        super().__init__()
        self.atlas_directory = atlas_directory
        self.month = month

        self.filename_pattern = "ssmi_mean_emis_climato_@MM@_cov_interpol_M2"
        self.r_min = 0.0
        self.r_max = 1.0
        self.d_max = -1.0
        self.name = "surface"


    @property
    def required_data(self):
        return [("temperature", ("n_lat", "n_lon"), False)]

    @arts_property(group="Vector",
                   shape=(Dimension.Lat,),
                   wsv=wsv["lat_true"])
    def latitude(self):
        return None

    @arts_property(group="Vector",
                  shape=(Dimension.Lon,),
                  wsv=wsv["lon_true"])
    def longitude(self):
        return None

    @arts_property(group="Matrix",
                   shape=(Dimension.Lat, Dimension.Lon),
                   wsv=wsv["t_surface"])
    def temperature(self):
        return None

    @property
    def surface_agenda(self):

        @arts_agenda
        def surface_rtprop_agenda_telsem(ws):
            ws.specular_losCalc()
            ws.InterpSurfaceFieldToPosition(out = wsv["surface_skin_t"],
                                            field = wsv["t_surface"])
            ws.surfaceTelsem(atlas=ws.telsem_atlas,
                             r_min=self.r_min,
                             r_max=self.r_max,
                             d_max=self.d_max)
        return surface_rtprop_agenda_telsem

    def setup(self, ws):
        ws.TelsemAtlasCreate("telsem_atlas")
        ws.telsem_atlasesReadAscii(directory=self.atlas_directory,
                                   filename_pattern=self.filename_pattern)
        ws.Extract(ws.telsem_atlas, wsv["telsem_atlases"], self.month)
        ws.Copy(ws.surface_rtprop_agenda, self.surface_agenda)

    def get_data(self, ws, data_provider, *args, **kwargs):
        self.get_data_arts_properties(ws, data_provider, *args, **kwargs)

    def run_checks(self, ws):
        pass
