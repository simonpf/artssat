"""
The PSD Submodule
=================

This submodule provides functionality for the handling of particle size
distributions (PSDs) used to describe particle ensembles in radiative
transfer simulations. It serves two purposes:

1. Provision of an abstract interface for the handling of PSDs in ARTS
2. Provision of functionality for the handling and conversion of PSD data

ARTS interface
--------------

The ARTS interface is implemented by the :class:`ArtsPSD` class, which
implements common functionality for the handling of PSDs in ARTS and defines
the interaction with the :class:`ArtsSimulation` and in particular
the :class:`ScatteringSpecies` instance in which the PSD is used.

.. toctree::
   :maxdepth: 1

   parts.scattering.psd.arts_psd

Handling of PSD data
--------------------

The :class:`PSDData` class handles numeric PSD data. Numeric PSD data
here refers to a PSD that is not given in a parametrized form but
instead by an array of particle density values over a given size grid.

.. toctree::
   :maxdepth: 1

   parts.scattering.psd.psd_data

PSD Classes
-----------

.. toctree::
   :maxdepth: 1

   parts.scattering.psd.d14

"""
from parts.scattering.psd.d14 import D14, D14N

