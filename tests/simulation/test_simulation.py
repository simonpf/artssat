import numpy as np

from parts import ArtsSimulation
from parts.atmosphere import Atmosphere1D
from parts.atmosphere.absorption import O2, N2, H2O
from parts.scattering import ScatteringSpecies, D14
from parts.atmosphere.surface import Tessem
from parts.sensor import ICI
from examples.data_provider import DataProvider, APrioriProvider, DataProvider2Ice

import matplotlib.pyplot as plt

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
    print(ici.y)

def test_simulation_scattering():

    scattering_data = "/home/simon/src/parts/tests/data/SectorSnowflake.xml"
    scattering_meta = "/home/simon/src/parts/tests/data/SectorSnowflake.meta.xml"
    ice = ScatteringSpecies("ice", D14(-1.0, 2.0),
                            scattering_data = scattering_data,
                            scattering_meta_data = scattering_meta)
    ice.psd.t_min = 0.0
    ice.psd.t_max = 275.0

    atmosphere = Atmosphere1D(absorbers = [O2(), N2(), H2O()],
                              scatterers = [ice],
                              surface = Tessem())
    ici = ICI()
    ici.sensor_line_of_sight = np.array([[135.0]])
    ici.sensor_position = np.array([[600e3]])

    simulation = ArtsSimulation(atmosphere = atmosphere,
                                data_provider = DataProvider(),
                                sensors = [ici])
    simulation.setup()
    simulation.run()
    print(simulation.workspace.y.value)

    simulation = ArtsSimulation(atmosphere = atmosphere,
                                data_provider = DataProvider2Ice(),
                                sensors = [ici])
    simulation.setup()
    simulation.run()
    print(simulation.workspace.y.value)
    return simulation.workspace

def test_simulation_scattering_jacobian():

    scattering_data = "/home/simon/src/parts/tests/data/SectorSnowflake.xml"
    scattering_meta = "/home/simon/src/parts/tests/data/SectorSnowflake.meta.xml"
    ice = ScatteringSpecies("ice", D14(-1.0, 2.0),
                            scattering_data = scattering_data,
                            scattering_meta_data = scattering_meta)
    ice.psd.t_min = 0.0
    ice.psd.t_max = 275.0

    atmosphere = Atmosphere1D(absorbers = [O2(), N2(), H2O()],
                              scatterers = [ice],
                              surface = Tessem())
    ici = ICI()
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

    scattering_data = "/home/simon/src/parts/tests/data/SectorSnowflake.xml"
    scattering_meta = "/home/simon/src/parts/tests/data/SectorSnowflake.meta.xml"
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
    ice.mass_density.limit_low = 0.0
    simulation.retrieval.settings["max_iter"] = 2

    simulation.retrieval.y = y
    simulation.setup()
    simulation.run()
    return simulation.workspace

def test_simulation_absorption_jacobian():
    atmosphere = Atmosphere1D(absorbers = [O2(), N2(), H2O()],
                              surface = Tessem())
    o2, n2, h2o = atmosphere.absorbers

    ici = ICI()
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

    print(simulation.workspace.x.value)
