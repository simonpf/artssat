"""

The :code:`artssat` package provides a modular framework for
radiative transfer simulations. It does so by defining a general
computing workflow that allows hiding away much of the complexity
of ARTS behind physically-motivated Python classes.

Overview
--------

The :class:`ArtsSimulation` class provides the main user interface
of the :code:`artssat` package. As the name suggests, it represents
an ARTS simulation, which may be a simple forward simulation, a
Jacobian calculation or a retrieval calculation.

The :code:`atmosphere` attribute of an :class:`ArtsSimulation` object
represents the model atmosphere that is underlying the radiative transfer
simulation. What kind of radiative transfer simulations are performed on the
model atmosphere is defined by the :code:`sensors` attribute of the simulation.
These are the main components that describe the general simulation *setup*.

It is important to note that a :class:`ArtsSimulation` object represents only
an abstract description of the simulation scenario. That means it doesn't hold
any data describing a specific atmospheric state, such as for example the
temperature field. The rationale for this design decision is to separate the
simulation setup from the data source, so that these two can be exchanged more
easily.

The data describing a concrete atmsopheric state to be simulated is expected
to be provided by a **data provider**. The data providers task is to provide
get methods for all data required for the simulation. These get methods may
be parametrized to allow the simulation of several different states. When
the :code:`run(*args, **kwargs)` method of the :class:`ArtsSimulation` is
called, it forwards its arguments to the getter methods of the data provider.
For a more detailed description of the data flow see the section below.

The image below illustrates the general semantic model used to describe a
radiative transfer simulation.

.. image:: /figures/overview_diagram.svg

Interaction and data workflow
-----------------------------

The general interaction and data flow for a single simulation is illustrated
in the figure below.

After defining the simulation setup, preparatory calculations are performed
by calling the :code:`setup()` method of the :class:`ArtsSimulation` object.
To then run a simulation, the user calls the :code:`run(...)` method providing
as arguments the arguments that should be passed on to the :code:`data_provider`
to return the data of the atmospheric state to simulate (Step 1).

Depending on the simulation setup, the :class:`ArtsSimulation` object will
require different data to perform the simulation. These quantities
are identified by a unique name :code:`name` and for each of these the
:class:`ArtsSimulation` object will call the corresponding :code:`get_<name>`
method of the :code:`data_provider` with the arguments provided by the
user to the :code:`run` method (Step 2).

After receiving the required data from the :code:`data_provider` (Step 3),
the :class:`ArtsSimulation` object performs the simulation and returns the
result to the user (Step 4).

.. image:: /figures/data_flow_diagram.svg

"""
import artssat.dimensions
from artssat.simulation import ArtsSimulation
