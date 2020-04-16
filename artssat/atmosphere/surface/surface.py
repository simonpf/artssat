"""
The surface sub-module provides implementations of the different surface models
that are available in ARTS.
"""
from abc import abstractmethod, abstractproperty
import numpy as np

from artssat.arts_object import ArtsObject, arts_property, Dimension
from pyarts.workspace import Workspace, arts_agenda
from pyarts.workspace.variables import (WorkspaceVariable, workspace_variables)
wsv = workspace_variables

ws = Workspace(verbosity = 0)


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
    def get_data(self, ws, data_provider, *args, **kwargs):
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

#TODO: Need elegant system for on demand WSVs.
telsem_salinity = ws.create_variable("Numeric", None)
telsem_windspeed = ws.create_variable("Numeric", None)

class Tessem(Surface,
             ArtsObject):
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
        Surface.__init__(self)
        ArtsObject.__init__(self)

        self.tessem_net_h = tessem_net_h
        self.tessem_net_v = tessem_net_v

    @property
    def required_data(self):
        return [("surface_temperature", ("n_lat", "n_lon"), False),
                ("surface_salinity", (1,), True),
                ("surface_salinity", ("n_lat", "n_lon"), True),
                ("surface_wind_speed", (1,), True),
                ("surface_wind_speed", ("n_lat", "n_lon"), True)]

    @arts_property(group="Matrix",
                   shape=(Dimension.Lat, Dimension.Lon),
                   wsv=wsv["t_surface"])
    def surface_temperature(self):
        return 280.0

    @arts_property(group="Numeric",
                   wsv=telsem_salinity)
    def salinity(self):
        return 0.035

    @arts_property(group="Numeric",
                   wsv=telsem_windspeed)
    def surface_wind_speed(self):
        return 1.0

    @property
    def surface_agenda(self):

        @arts_agenda
        def surface_rtprop_agenda_tessem(ws):
            ws.specular_losCalc()
            ws.InterpSurfaceFieldToPosition(out = wsv["surface_skin_t"],
                                            field = wsv["t_surface"])
            ws.surfaceTessem(salinity = telsem_salinity,
                                wind_speed = telsem_windspeed)
        return surface_rtprop_agenda_tessem

    def setup(self, ws):
        ws.TessemNNReadAscii(ws.tessem_neth, self.tessem_net_h)
        ws.TessemNNReadAscii(ws.tessem_netv, self.tessem_net_v)
        ws.Copy(ws.surface_rtprop_agenda, self.surface_agenda)

    def get_data(self, ws, data_provider, *args, **kwargs):
        self.get_data_arts_properties(ws, data_provider, *args, **kwargs)

    def run_checks(self, ws):
        pass

################################################################################
# Telsem
################################################################################

class Telsem(Surface,
             ArtsObject):
    """
    This class implements the Tool for estimating surface emissivities 2
    (TELSEM2). TELSEM2 is a microwave atlas covering microwave and
    sub-millimeter frequencies. It comes in the form of 12 microwave atlases,
    one for each month that contain microwave emissivities of land-surfaces on
    an equal-area grid.
    """
    def __init__(self,
                 atlas_directory,
                 month=6):
        """
        Args:
            atlas_directory: Directory containing the Telsem atlases
            month: The month of the atlas to use.
        """
        Surface.__init__(self)
        ArtsObject.__init__(self)

        self.atlas_directory = atlas_directory
        self.month = month

        self.filename_pattern = "ssmi_mean_emis_climato_@MM@_cov_interpol_M2"
        self.r_min = 0.0
        self.r_max = 1.0
        self.d_max = -1.0

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
    def surface_temperature(self):
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

################################################################################
# Combined surface
################################################################################

class CombinedSurface(ArtsObject):
    """
    This class combines two surface models and switches between them depending
    on a surface property variable provided by the data provider.
    """
    def __init__(self,
                 surface_model_1,
                 surface_model_2,
                 surface_type_variable = "surface_type"):
        """
        Args:
            surface_model_1: The surface model to use when surface type is 0
            surface_model_2: The surface model to use when surface type is 1
            surface_type_variable: The variable based on which to switch between
                 the two models.
        """
        super().__init__()
        self.surface_model_1 = surface_model_1
        self.surface_model_2 = surface_model_2
        self.surface_type_variable = surface_type_variable

    @property
    def required_data(self):
        return self.surface_model_1.required_data + self.surface_model_2.required_data

    @arts_property(group="Index")
    def surface_type(self):
        return None

    @property
    def surface_agenda(self):

        @arts_agenda
        def surface_agenda(ws):
            ws.Ignore(ws.f_grid)
            ws.Ignore(ws.rtp_pos)
            ws.Ignore(ws.rtp_los)
            ws.Touch(ws.surface_skin_t)
            ws.Touch(ws.surface_emission)
            ws.Touch(ws.surface_los)
            ws.Touch(ws.surface_rmatrix)
            if (self.surface_type == 0):
                ws.execute_agenda(self.surface_agenda_1)
            else:
                ws.execute_agenda(self.surface_agenda_2)

        return surface_agenda

    def setup(self, ws):
        self.surface_model_1.setup(ws)
        self.surface_model_2.setup(ws)
        ws.Copy(ws.surface_rtprop_agenda, self.surface_agenda)


    def get_data(self, ws, data_provider, *args, **kwargs):
        self.surface_model_1.get_data(ws, data_provider, *args, **kwargs)
        self.surface_model_2.get_data(ws, data_provider, *args, **kwargs)
        self.surface_agenda_1 = self.surface_model_1.surface_agenda
        self.surface_agenda_2 = self.surface_model_2.surface_agenda
        self.get_data_arts_properties(ws, data_provider, *args, **kwargs)

        if self.surface_type <= 0.0:
            ws.Copy(ws.surface_rtprop_agenda, self.surface_model_1.surface_agenda)
        else:
            ws.Copy(ws.surface_rtprop_agenda, self.surface_model_2.surface_agenda)
        ws.Copy(ws.surface_rtprop_agenda, self.surface_agenda)

    def run_checks(self, ws):
        pass
