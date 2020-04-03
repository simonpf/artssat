"""
Binned
======

The binned PSD represents a discretized PSD that is represented
by particle densities over a given number of bins.
"""
import numpy as np
import scipy as sp
from pyarts.workspace import arts_agenda

from artssat.scattering.psd.arts.arts_psd import ArtsPSD
from artssat.scattering.psd.data.psd_data import PSDData, D_eq

class Binned(ArtsPSD):
    """
    Discrete representation of a PSD using fixed bins.
    """

    def convert_from(self, psd):
        """
        Convert another psd to a Binned object.
        """
        self.size_parameter = psd.size_parameter
        y = psd.evaluate(self.x).data
        inds = (slice(None),) * (len(y.shape) - 1)
        self.moments = [y[inds + (i,)] for i in range(self.x.size)]

    def __init__(self,
                 x,
                 size_parameter = D_eq(1000.0),
                 t_min = 0.0,
                 t_max = 400.0):
        """
        Arguments:

            x: Vector holding the positions of the centers of the size bins.

            size_parameter: The size parameter to use.
        """
        super().__init__(size_parameter)
        self.x = x.ravel()
        self.t_min = t_min
        self.t_max = t_max


    def get_moment(self, p, reference_size_parameter = None):
        if not reference_size_parameter is None:
            a1 = self.size_parameter.a
            b1 = self.size_parameter.b
            a2 = reference_size_parameter.a
            b2 = reference_size_parameter.b

            c = (a1 / a2) ** (p / b2)
            p = p * b1 / b2
        else:
            c = 1.0

        data = np.array([m for m in self.moments]).T
        x = np.broadcast_to(self.x.reshape(1, -1), data.shape)
        return c * np.trapz(data * x ** p, x = x)

    @property
    def moment_names(self):
        """
        Names of the moments of the PSD.

        These are just :code:`bin_0, bin_1, ...` for each bin
        of the PSD.
        """
        return ["bin_" + str(i) for i in range(self.x.size)]

    @property
    def pnd_call_agenda(self):
        @arts_agenda
        def pnd_call(ws):
            ws.Ignore(ws.pnd_agenda_input)
            ws.Ignore(ws.pnd_agenda_input_t)
            ws.Ignore(ws.pnd_agenda_input_names)
            ws.Ignore(ws.dpnd_data_dx_names)
            ws.Touch(ws.psd_data)
            ws.Touch(ws.dpsd_data_dx)

            ws.scat_species_a = self.size_parameter.a
            ws.scat_species_b = self.size_parameter.b

            xi = ws.psd_size_grid.value
            y  = ws.pnd_agenda_input.value
            y  = np.maximum(y, 1e-12)
            yi = sp.interpolate.interp1d(self.x.ravel(),
                                         np.log10(y), axis = 1, fill_value = 0.0)(xi)
            yi = 10.0 ** yi
            yi[yi == 1e-12] == 0.0

            t = ws.pnd_agenda_input_t.value
            yi[t < self.t_min, :] = 0.0
            yi[t > self.t_max, :] = 0.0
            yi = np.maximum(yi, 0.0)

            ws.psd_data.value = np.copy(yi)
            ndx = len(ws.dpnd_data_dx_names.value)
            ws.dpsd_data_dx = np.ones((ndx, yi.shape[0], self.x.size))

        return pnd_call
