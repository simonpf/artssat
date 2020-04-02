"""
Setup routines to simplify simulation setup.
"""
import parts
from parts.scattering.solvers import RT4, Disort
from parts                    import ArtsSimulation
from parts.models             import StandardAtmosphere
from parts.data.atmosphere    import Tropical
from parts.atmosphere         import Atmosphere1D
from parts.sensor             import ICI, MWI
from parts.scattering         import ScatteringSpecies, D14
from parts.data_provider      import FunctorDataProvider
from parts.retrieval.a_priori import FixedAPriori, Diagonal, \
    SensorNoiseAPriori

import pyarts
from pyarts.workspace.api import include_path_push

import numpy as np

import os
import sys
base_path = os.path.join(os.path.dirname(parts.__file__), "..")
sys.path.append(base_path)
sys.path.append(os.path.join(base_path, "tests"))
from utils.data import scattering_data, scattering_meta

include_path_push(os.path.join(test_path, "tests", "data"))

def arts_simulation(scattering = False):

    # Sensors
    ici = ICI(channel_indices = [0, -1], stokes_dimension = 1)
    ici.sensor_position       = 500e3
    ici.sensor_line_of_sight  = 180.0
    mwi = MWI(channel_indices = [-2, -1], stokes_dimension = 1)
    mwi.sensor_position       = 500e3
    mwi.sensor_line_of_sight  = 180.0
    sensors       = [ici, mwi]

    # Atmosphere
    atmosphere    = StandardAtmosphere()
    if scattering:
        ice = ScatteringSpecies("ice", D14(-1.0, 2.0),
                                scattering_data = scattering_data,
                                scattering_meta_data = scattering_meta)
        ice.psd.t_min = 0.0
        ice.psd.t_max = 275.0
        atmosphere.scatterers = [ice]

    data_provider = Tropical()
    if scattering:
        def gaussian_cloud_1(z):
            return 1e-4 * np.exp(- ((z - 12e-3) / 1e3) ** 2.0)
        name = ice.name + "_" + ice.psd.moment_names[0]
        data_provider.add(FunctorDataProvider(name, "altitude", gaussian_cloud_1))

        def gaussian_cloud_2(z):
            return 1e-4 * np.ones(z.shape)
        name = ice.name + "_" + ice.psd.moment_names[1]
        data_provider.add(FunctorDataProvider(name, "altitude", gaussian_cloud_2))


    simulation    = ArtsSimulation(atmosphere = atmosphere,
                                   sensors = sensors,
                                   data_provider = data_provider)
    return simulation

def arts_retrieval(scattering = False):
    """
    Setup up a retrieval simulation with or without
    scattering.
    """

    simulation = arts_simulation(scattering = scattering)

    if scattering:
        q = simulation.atmosphere.scatterers[0].moments[0]
        simulation.retrieval.add(q)
        dp = FixedAPriori("ice_mass_density", 1e-4, Diagonal(1.0))
        simulation.data_provider.add(dp)
    else:
        q = simulation.atmosphere.absorbers[0]
        simulation.retrieval.add(q)
        dp = FixedAPriori("O2", 0.3, Diagonal(0.005))
        simulation.data_provider.add(dp)
    simulation.data_provider.add(SensorNoiseAPriori(simulation.sensors))
    return simulation
