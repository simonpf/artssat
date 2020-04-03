from pyarts.workspace import Workspace, arts_agenda
from abc import abstractmethod, abstractproperty
import numpy as np

class Surface:

    @abstractproperty
    def required_data(self):
        pass

    @abstractmethod
    def setup(self, ws):
        pass

    @abstractmethod
    def get_data(self, ws, provider, *args, **kwargs):
        pass

    @abstractmethod
    def run_checks(self):
        pass

class Tessem(Surface):

    def __init__(self,
                 tessem_net_h = "testdata/tessem_sav_net_V.txt",
                 tessem_net_v = "testdata/tessem_sav_net_V.txt",
                 salinity = 0.034,
                 wind_speed = 0.0):
        self.tessem_net_h = tessem_net_h
        self.tessem_net_v = tessem_net_v

        ws = Workspace()

        self._surface_temperature_field = ws.add_variable(np.zeros((1, 1)))

        self._surface_salinity       = ws.add_variable(salinity)
        self._surface_salinity_field = ws.add_variable(np.zeros((1, 1)))

        self._surface_wind_speed       = ws.add_variable(0.0)
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
