import numpy as np
import pytest

from parts import ArtsSimulation
from parts.atmosphere import Atmosphere1D
from parts.atmosphere.absorption import O2, N2, H2O
from parts.scattering import ScatteringSpecies, D14
from parts.scattering.solvers import RT4, Disort
from parts.atmosphere.surface import Tessem
from parts.sensor import CloudSat, ICI

from examples.data_provider import DataProvider, \
    APrioriProvider, APrioriProviderCombined
from tests.data import scattering_data, scattering_meta

import matplotlib.pyplot as plt

scattering_solvers = pytest.mark.parametrize("scattering_solver", [RT4, Disort])

#from IPython import get_ipython
#ip = get_ipython()
#ip.magic("%load_ext autoreload")
#ip.magic("%autoreload 2")
import os

def test_simulation_absorption():
    atmosphere = Atmosphere1D(absorbers = [O2(), N2(), H2O()],
                              surface = Tessem())
    ici = ICI()
    ici.sensor_line_of_sight = np.array([[135.0]])
    ici.sensor_position = np.array([[600e3]])

    simulation = ArtsSimulation(atmosphere = atmosphere,
                                data_provider = DataProvider(),
                                sensors = [ici])
    simulation.setup()
    simulation.run()

    y1 = np.copy(ici.y)

    simulation.run()
    y2 = np.copy(ici.y)

    assert(np.all(np.isclose(y1, y2)))

@scattering_solvers
def test_simulation_scattering(scattering_solver):

    ice = ScatteringSpecies("ice", D14(-1.0, 2.0),
                            scattering_data = scattering_data,
                            scattering_meta_data = scattering_meta)
    ice.psd.t_min = 0.0
    ice.psd.t_max = 275.0

    atmosphere = Atmosphere1D(absorbers = [O2(), N2(), H2O()],
                              scatterers = [ice],
                              surface = Tessem())
    ici = ICI(stokes_dimension = 1, channels = [1, -1])
    ici.sensor_line_of_sight = np.array([[135.0]])
    ici.sensor_position = np.array([[600e3]])

    simulation = ArtsSimulation(atmosphere = atmosphere,
                                data_provider = DataProvider(),
                                sensors = [ici])
    simulation.scattering_solver = scattering_solver()
    simulation.setup()
    simulation.run()

def test_simulation_scattering_jacobian():

    ice = ScatteringSpecies("ice", D14(-1.0, 2.0),
                            scattering_data = scattering_data,
                            scattering_meta_data = scattering_meta)
    ice.psd.t_min = 0.0
    ice.psd.t_max = 275.0

    atmosphere = Atmosphere1D(absorbers = [O2(), N2(), H2O()],
                              scatterers = [ice],
                              surface = Tessem())
    ici = ICI(channels = [1, -1])
    ici.sensor_line_of_sight = np.array([[135.0]])
    ici.sensor_position = np.array([[600e3]])

    simulation = ArtsSimulation(atmosphere = atmosphere,
                                data_provider = DataProvider(),
                                sensors = [ici])
    simulation.jacobian.add(ice.mass_density)
    simulation.setup()
    simulation.run()
    print(simulation.workspace.particle_bulkprop_field.value)
    return np.copy(simulation.workspace.jacobian.value)

def test_simulation_scattering_retrieval():

    ice = ScatteringSpecies("ice", D14(-1.0, 2.0),
                            scattering_data = scattering_data,
                            scattering_meta_data = scattering_meta)
    ice.psd.t_min = 0.0
    ice.psd.t_max = 275.0

    ici = ICI()
    ici.sensor_line_of_sight = np.array([[135.0]])
    ici.sensor_position = np.array([[600e3]])


    atmosphere = Atmosphere1D(absorbers = [O2(), N2(), H2O()],
                              scatterers = [ice],
                              surface = Tessem())
    simulation = ArtsSimulation(atmosphere = atmosphere,
                                data_provider = APrioriProvider(),
                                sensors = [ici])

    simulation.setup()
    simulation.run()
    y = np.copy(simulation.workspace.y)

    simulation.retrieval.add(ice.mass_density)
    ice.mass_density.limit_low = 1e-6
    simulation.retrieval.settings["max_iter"] = 5

    simulation.retrieval.y = y
    simulation.setup()
    simulation.run()
    return simulation.workspace

@scattering_solvers
def test_simulation_scattering_combined(scattering_solver):

    ice = ScatteringSpecies("ice", D14(-1.0, 2.0),
                            scattering_data = scattering_data,
                            scattering_meta_data = scattering_meta)
    ice.psd.t_min = 0.0
    ice.psd.t_max = 275.0

    ici = ICI(channels = [0, -1], stokes_dimension = 1)
    ici.sensor_line_of_sight = np.array([[135.0]])
    ici.sensor_position = np.array([[600e3]])

    cs = CloudSat(stokes_dimension = 1)
    cs.range_bins = np.linspace(0, 30e3, 31)
    cs.sensor_line_of_sight = np.array([[135.0]])
    cs.sensor_position = np.array([[600e3]])

    atmosphere = Atmosphere1D(absorbers = [O2(), N2(), H2O()],
                              scatterers = [ice],
                              surface = Tessem())
    simulation = ArtsSimulation(atmosphere = atmosphere,
                                data_provider = APrioriProvider(),
                                sensors = [ici, cs])

    simulation.scattering_solver = scattering_solver()
    simulation.setup()
    simulation.run()
    y = np.copy(simulation.workspace.y)
    return y

def test_simulation_scattering_combined_retrieval():

    ice = ScatteringSpecies("ice", D14(-1.0, 2.0),
                            scattering_data = scattering_data,
                            scattering_meta_data = scattering_meta)
    ice.psd.t_min = 0.0
    ice.psd.t_max = 275.0

    ici = ICI()
    ici.sensor_line_of_sight = np.array([[135.0]])
    ici.sensor_position = np.array([[600e3]])

    cs = CloudSat()
    cs.range_bins = np.linspace(0, 30e3, 31)
    cs.sensor_line_of_sight = np.array([[135.0]])
    cs.sensor_position = np.array([[600e3]])

    atmosphere = Atmosphere1D(absorbers = [O2(), N2(), H2O()],
                              scatterers = [ice],
                              surface = Tessem())
    simulation = ArtsSimulation(atmosphere = atmosphere,
                                data_provider = APrioriProviderCombined(),
                                sensors = [ici, cs])

    simulation.setup()
    simulation.run()
    y = np.copy(simulation.workspace.y)

    simulation.retrieval.add(ice.mass_density)
    simulation.retrieval.add(ice.mass_weighted_diameter)
    ice.mass_density.retrieval.limit_low = 1e-10
    ice.mass_weighted_diameter.retrieval.limit_low = 1e-5
    simulation.retrieval.settings["max_iter"] = 10
    simulation.retrieval.y = y

    simulation.setup()
    simulation.run()
    y_f = np.copy(simulation.workspace.yf.value)
    x   = np.copy(simulation.workspace.x.value)
    #return x, y_f, y
    return simulation.workspace

def test_simulation_absorption_jacobian():
    atmosphere = Atmosphere1D(absorbers = [O2(), N2(), H2O()],
                              surface = Tessem())
    o2, n2, h2o = atmosphere.absorbers

    ici = ICI(channels = [0, -1])
    ici.sensor_line_of_sight = np.array([[135.0]])
    ici.sensor_position = np.array([[600e3]])

    simulation = ArtsSimulation(atmosphere = atmosphere,
                                data_provider = DataProvider(),
                                sensors = [ici])
    simulation.jacobian.add(o2)
    simulation.jacobian.add(n2)
    simulation.jacobian.add(h2o)
    simulation.setup()
    simulation.run()
    return np.copy(simulation.workspace.jacobian.value)

def test_simulation_absorption_retrieval():
    atmosphere = Atmosphere1D(absorbers = [O2(), N2(), H2O()],
                              surface = Tessem())
    o2, n2, h2o = atmosphere.absorbers

    ici = ICI()
    ici.sensor_line_of_sight = np.array([[135.0]])
    ici.sensor_position = np.array([[600e3]])

    simulation = ArtsSimulation(atmosphere = atmosphere,
                                data_provider = APrioriProvider(),
                                sensors = [ici])

    simulation.setup()
    simulation.run()
    y = np.copy(simulation.workspace.y)

    simulation.retrieval.add(o2)
    simulation.retrieval.y = y

    simulation.setup()
    simulation.run()
