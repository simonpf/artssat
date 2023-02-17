import numpy as np
import pytest
import os
import sys

import artssat
test_path = os.path.join(os.path.dirname(artssat.__file__), "..", "tests")
sys.path.append(test_path)
from artssat import ArtsSimulation
from artssat.atmosphere import Atmosphere1D
from artssat.atmosphere.absorption import O2, N2, H2O
from artssat.scattering import ScatteringSpecies, D14
from artssat.scattering.solvers import RT4, Disort
from artssat.atmosphere.surface import Tessem
from artssat.sensor import CloudSat, ICI
from examples.data_provider import DataProvider
from utils.data import scattering_data, scattering_meta

################################################################################
# Absorption only
################################################################################

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

################################################################################
# Absorption and scattering
################################################################################

scattering_solvers = pytest.mark.parametrize("scattering_solver", [RT4, Disort])
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
    ici = ICI(stokes_dimension = 1, channel_indices = [1, -1])
    ici.sensor_line_of_sight = np.array([[135.0]])
    ici.sensor_position = np.array([[600e3]])

    simulation = ArtsSimulation(atmosphere = atmosphere,
                                data_provider = DataProvider(),
                                sensors = [ici])
    simulation.scattering_solver = scattering_solver()
    simulation.setup()
    simulation.run()

################################################################################
# Scattering + Jacobian
################################################################################

def test_simulation_scattering_jacobian():

    ice = ScatteringSpecies("ice", D14(-1.0, 2.0),
                            scattering_data = scattering_data,
                            scattering_meta_data = scattering_meta)
    ice.psd.t_min = 0.0
    ice.psd.t_max = 275.0

    atmosphere = Atmosphere1D(absorbers = [O2(), N2(), H2O()],
                              scatterers = [ice],
                              surface = Tessem())
    ici = ICI(channel_indices = [1, -1])
    ici.sensor_line_of_sight = np.array([[135.0]])
    ici.sensor_position = np.array([[600e3]])

    simulation = ArtsSimulation(atmosphere = atmosphere,
                                data_provider = DataProvider(),
                                sensors = [ici])
    simulation.jacobian.add(ice.mass_density)
    simulation.setup()
    simulation.run()
    print(simulation.workspace.particle_bulkprop_field.value)

################################################################################
# Active + Passive, Jacobian
################################################################################

@scattering_solvers
def test_simulation_scattering_combined(scattering_solver):

    ice = ScatteringSpecies("ice", D14(-1.0, 2.0),
                            scattering_data = scattering_data,
                            scattering_meta_data = scattering_meta)
    ice.psd.t_min = 0.0
    ice.psd.t_max = 275.0

    ici = ICI(channel_indices = [0, -1], stokes_dimension = 1)
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
                                data_provider = DataProvider(),
                                sensors = [ici, cs])

    simulation.scattering_solver = scattering_solver()
    simulation.setup()
    simulation.run()
    y = np.copy(simulation.workspace.y)

################################################################################
# Absorption + Jacobian
################################################################################

def test_simulation_absorption_jacobian():
    atmosphere = Atmosphere1D(absorbers = [O2(), N2(), H2O()],
                              surface = Tessem())
    o2, n2, h2o = atmosphere.absorbers

    ici = ICI(channel_indices = [0, -1])
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

################################################################################
# Multiple views
################################################################################

def test_simulation_multiview():
    atmosphere = Atmosphere1D(absorbers = [O2(), N2(), H2O()],
                              surface = Tessem())
    lines_of_sight = np.array([[135.0],
                               [180.0]])
    positions = np.array([[600e3],
                          [600e3]])

    ici = ICI(lines_of_sight=lines_of_sight,
              positions=positions)

    simulation = ArtsSimulation(atmosphere = atmosphere,
                                data_provider = DataProvider(),
                                sensors = [ici])
    simulation.setup()
    simulation.run()

    y = np.copy(ici.y)
    assert(y.shape[0] == 2)

def test_simulation_multiview_radar():
    atmosphere = Atmosphere1D(absorbers = [O2(), N2(), H2O()],
                              surface = Tessem())
    lines_of_sight = np.array([[135.0],
                               [180.0]])
    positions = np.array([[600e3],
                          [600e3]])

    cs = CloudSat(lines_of_sight=lines_of_sight,
                   positions=positions)

    simulation = ArtsSimulation(atmosphere = atmosphere,
                                data_provider = DataProvider(),
                                sensors = [cs])
    simulation.setup()
    simulation.run()

    y = np.copy(cs.y)
    assert(y.shape[0] == 2)
    assert(y.shape[-1] == 2)


ice = ScatteringSpecies("ice", D14(-1.0, 2.0),
                        scattering_data = scattering_data,
                        scattering_meta_data = scattering_meta)
ice.psd.t_min = 0.0
ice.psd.t_max = 275.0

atmosphere = Atmosphere1D(absorbers = [O2(), N2(), H2O()],
                            scatterers = [],
                            surface = Tessem())
ici = ICI(stokes_dimension = 1, channel_indices = [1, -1])
ici.sensor_line_of_sight = np.array([[135.0]])
ici.sensor_position = np.array([[600e3]])

simulation = ArtsSimulation(atmosphere = atmosphere,
                            data_provider = DataProvider(),
                            sensors = [ici])
simulation.scattering_solver = RT4()
simulation.setup()
for i in range(100):
    simulation.run()
