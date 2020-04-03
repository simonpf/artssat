"""
Fixed shape PSD
===============

The fixed-shape PSD takes a fixed PSD shape and scales it with
a given mass density.
"""
from artssat.scattering.psd.arts.arts_psd import ArtsPSD
from artssat.scattering.psd.data.psd_data import PSDData, D_eq
from artssat.arts_object import arts_property
from artssat.arts_object import Dimension as dim
from artssat.arts_object import ArtsObject
from pyarts.workspace import Workspace, arts_agenda

ws = Workspace()

class FixedShape(ArtsPSD, ArtsObject):
    """
    Fixed shape particle size distribution.

    This PSD class takes a mass density as input and returns a given PSD shape
    scaled in the vertical dimension to match the provided mass density.
    """
    @arts_property("Vector")
    def x(self):
        return None

    @arts_property("Matrix")
    def shape(self):
        return None

    def __init__(self, x, data, size_parameter = D_eq(1000.0)):
        """
        Create PSD with shape given by :code:`x` and :code:`data`.

        Arguments:
            x(numpy.ndarray): 1D array containing the centers of the size
                bins of the provided PSD shape.

            data(numpy.ndarray): 1D array containing the PSD representing the
                shape of the PSD.

            size_parameter: :class:`SizeParameter` representing the size parameter
                over which the PSD is defined.
        """
        ArtsObject.__init__(self)
        ArtsPSD.__init__(self, self.psd.size_parameter)
        self.psd  = PSDData(x, data, size_parameter)
        self.x    = x
        self.data = data

        shape = self.psd.data.reshape((1, -1))
        self.shape = shape / self.psd.get_mass_density()
        self.size_parameter = self.psd.size_parameter

        self.name = "fixed_psd"

    @property
    def moment_names(self):
        return ["mass_density"]

    @property
    def pnd_call_agenda(self):
        @arts_agenda
        def pnd_call(ws):
            ws.Ignore(ws.pnd_agenda_input_t)
            ws.Ignore(ws.pnd_agenda_input)
            ws.Ignore(ws.pnd_agenda_input_names)
            ws.Ignore(ws.dpnd_data_dx_names)
            ws.Touch(ws.psd_data)
            ws.Touch(ws.dpsd_data_dx)

            md = ws.pnd_agenda_input.value.reshape(-1, 1)
            ws.pnd_size_grid = self.x
            ws.psd_size_grid = self.x
            ws.psd_data    = self.shape * md
            ws.psd_data_dx = self.shape

        return pnd_call

    def setup(self, ws, i):
        self.pbf_index = i

    def get_data(self, ws, i, *args, **kwargs):
        pass
