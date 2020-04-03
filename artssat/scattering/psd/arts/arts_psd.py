"""
The ArtsPSD class
=================

"""

import numpy as np
import scipy as sp
from abc import abstractproperty
from artssat.scattering.psd.data.psd_data import SizeParameter, Area, D_eq,\
    D_max, Mass
from pyarts.workspace import arts_agenda

class ArtsPSD:
    r"""

    The interface for ARTS PSDs.

    In ARTS, A PSD for a given scattering property is defined by the
    corresponding agenda in the :code:`pnd_agenda_array`. The task
    of the :class:`ArtsPSD` class is to provide this agenda together
    with the names of the moments of the distribution.

    Attributes:

        size_parameter_names(dict): Dictionary that translates the
            :class:`SizeParameterEnum` into the corresponding name of
            the size parameter in ARTS.

    """

    properties = [("size_parameter", (), SizeParameter),
                  ("t_min", (), np.float),
                  ("t_max", (), np.float),
                  ("x_fit_start", (), np.float)]

    size_parameter_names = {D_eq : "dveq",
                            D_max : "dmax",
                            Mass : "mass",
                            Area : "area"}

    def __init__(self,
                 size_parameter,
                 t_min = 0.0,
                 t_max = 999.0):
        """
        Parameters:
            size_parameter(SizeParameter) :class:`SizeParameter` enum specifying the
                size parameter that is used by the PSD.
            t_min(numpy.float): ARTS parameter, minimum temperature for which PSD values
                will be produced.
            m_max(numpy.float): ARTS parameter, maximum temperature for which PSD values
                will be produced.
        """
        self.size_parameter = size_parameter
        self.t_min = t_min
        self.t_max = t_max

        self.x_fit_start = 100e-6

    #
    # Abstract properties
    #

    @abstractproperty
    def pnd_call_agenda(self):
        """
        The WSM call that is used to compute the PSD in ARTS.
        """
        pass

    @abstractproperty
    def moment_names(self):
        """
        List of strings that contains the names of the moments of the PSD.
        This is used in the ARTS :code:`particle_bulkrprop_names` and as
        name for the data requested from the data provider.
        """
        pass

    #
    # Generic PND agenda
    #

    @property
    def agenda(self):
        """
        The ARTS agenda representing the PSD. Should be used as :code:`pnd_agenda`
        in the ARTS :code:`pnd_agenda_array`.
        """
        size_parameter = ArtsPSD.size_parameter_names[type(self.size_parameter)]

        @arts_agenda
        def pnd_agenda(ws):
            ws.Ignore(ws.pnd_agenda_input_t)
            ws.Ignore(ws.pnd_agenda_input)
            ws.Ignore(ws.pnd_agenda_input_names)
            ws.Ignore(ws.dpnd_data_dx_names)
            ws.ScatSpeciesSizeMassInfo(species_index = ws.agenda_array_index,
                                       x_unit = size_parameter,
                                       x_fit_start = self.x_fit_start)
            ws.Copy(ws.psd_size_grid, ws.scat_species_x)
            ws.Copy(ws.pnd_size_grid, ws.scat_species_x)
            INCLUDE(self.pnd_call_agenda)
            ws.pndFromPsdBasic()

        return pnd_agenda
