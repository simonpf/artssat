import numpy as np

from parts import ArtsSimulation
from parts.atmosphere import Atmosphere1D
from parts.atmosphere.absorption import O2, N2, H2O
from parts.atmosphere.surface import Tessem
from parts.sensor import ICI
from examples.data_provider import DataProvider, APrioriProvider

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
    return simulation.workspace.jacobian.value

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
