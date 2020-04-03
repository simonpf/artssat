"""
artssat.models
==============

This module contains preconfigured :class:`artssat.ArtsSimulation` objects
that provide a starting point for performing radiative transfer calculations.
"""
from artssat.atmosphere.atmosphere import Atmosphere
from artssat.atmosphere.absorption import O2, N2, H2O
from artssat.atmosphere.surface    import Tessem

class StandardAtmosphere(Atmosphere):
    def __init__(self,
                 dimensions = 1,
                 surface = "ocean"):

        if not dimensions in [1]:
            raise Exception("Currently only 1D simulations are supported.")

        if not surface == "ocean":
            raise Exception("Currently only simulations over ocean surfaces are supported.")


        super().__init__(dimensions = (0, ) * dimensions,
                         absorbers = [O2(), N2(), H2O()],
                         scatterers = [],
                         surface = Tessem())



