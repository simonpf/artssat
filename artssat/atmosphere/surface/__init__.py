"""
Surface
=======

"""
import numpy as np
from pyarts.workspace import Workspace, arts_agenda

ws = Workspace()
ws.NumericCreate("surface_temperature")
ws.NumericCreate("surface_salinity")
ws.NumericCreate("surface_windspeed")

from artssat.atmosphere.surface.surface import Tessem, Telsem, CombinedSurface
