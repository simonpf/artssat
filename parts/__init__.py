"""
parts
=====

The :code:`parts` package provides a modular framework for
radiative transfer simulations. It does so by defining a general
computing workflow that allows hiding away much of the complexity
of ARTS behind physically-motivated Python classes.

Overview
--------

The :class:`ArtsSimulation` class provides the main user interface
of the :code:`parts` package. As the name suggests, it represents
an ARTS simulation, which may be a simple forward simulation, a
Jacobian calculation or a retrieval calculation.

The :class:`ArtsSimulation``s :code:`atmosphere` attribute represents
the atmosphere model that is underlying the radiative transfer
simulation. What kind of radiative transfer simulations are performed
on the model atmosphere is defined by the :code:`sensors` attributes
of the simulation. These are the main components that describe the
general simulation *setup*.

.. image:: /figures/overview_diagram.svg


"""
import parts.dimensions
from parts.simulation import ArtsSimulation
