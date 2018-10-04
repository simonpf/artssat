"""
The :code:`scattering` subpackage provides classes of scattering species
and particle size distributions (PSDs) for the simulation of scattering
in an ARTS simulation.

In addition to that, the :code: `parts.scattering.psd` subpackage
contains some functionality for the handling of PSD data.

"""
from parts.scattering.scattering_species \
    import ScatteringSpecies
from parts.scattering.psd \
    import D14, MY05
from parts.scattering.solvers import RT4

