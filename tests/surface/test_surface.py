import numpy as np
import pytest

import artssat
from artssat import ArtsSimulation
from artssat.atmosphere import Atmosphere1D
from artssat.atmosphere.absorption import O2, N2, H2O
from artssat.scattering import ScatteringSpecies, D14
from artssat.atmosphere.surface import Tessem, Telsem, CombinedSurface
from artssat.sensor import CloudSat, MWI
from pyarts.workspace.api import include_path_push

include_path_push("../data")

from examples.data_provider import DataProvider

import matplotlib.pyplot as plt

import os
import sys
test_path = os.path.join(os.path.dirname(artssat.__file__), "..", "tests")
sys.path.append(test_path)
from utils.data import scattering_data, scattering_meta


def test_simulation_tessem():
    atmosphere = Atmosphere1D(absorbers = [O2(), N2(), H2O()],
                              surface = Tessem())
    mwi = MWI()
    mwi.sensor_line_of_sight = np.array([[135.0]])
    mwi.sensor_position = np.array([[600e3]])

    simulation = ArtsSimulation(atmosphere = atmosphere,
                                data_provider = DataProvider(),
                                sensors = [mwi])
    simulation.setup()
    simulation.run()

    y1 = np.copy(mwi.y)

    simulation.run()
    y2 = np.copy(mwi.y)

    assert(np.all(np.isclose(y1, y2)))


@pytest.mark.skip(reason="Needs TELSEM2 atlases.")
def test_simulation_telsem():
    atmosphere = Atmosphere1D(absorbers = [O2(), N2(), H2O()],
                              surface = Telsem("/home/simon/Downloads"))
    mwi = MWI()
    mwi.sensor_line_of_sight = np.array([[135.0]])
    mwi.sensor_position = np.array([[600e3]])

    data_provider = DataProvider()
    data_provider.surface_temperature = 280.0 * np.ones((1, 1))
    data_provider.latitude = 58.0
    data_provider.longitude = 12.0
    simulation = ArtsSimulation(atmosphere = atmosphere,
                                data_provider = data_provider,
                                sensors = [mwi])
    simulation.setup()
    simulation.run()

@pytest.mark.skip(reason="Needs TELSEM2 atlases.")
def test_simulation_combined():
    surface_1 = Tessem()
    surface_2 = Telsem("/home/simon/Downloads")
    surface = CombinedSurface(surface_1,
                              surface_2)
    atmosphere = Atmosphere1D(absorbers = [O2(), N2(), H2O()],
                              surface = surface)
    mwi = MWI()
    mwi.sensor_line_of_sight = np.array([[135.0]])
    mwi.sensor_position = np.array([[600e3]])

    data_provider = DataProvider()
    data_provider.surface_temperature = 280.0 * np.ones((1, 1))
    data_provider.surface_latitude = 58.0
    data_provider.surface_longitude = 12.0
    data_provider.surface_type = 0.0

    simulation = ArtsSimulation(atmosphere = atmosphere,
                                data_provider = data_provider,
                                sensors = [mwi])

    simulation.setup()
    simulation.run()
    y = np.copy(mwi.y)

    atmosphere_r = Atmosphere1D(absorbers = [O2(), N2(), H2O()],
                                surface = surface_1)
    simulation_r = ArtsSimulation(atmosphere = atmosphere_r,
                                  data_provider = data_provider,
                                  sensors = [mwi])
    simulation_r.setup()
    simulation_r.run()
    y_r = np.copy(mwi.y)

    assert(np.allclose(y, y_r))

    data_provider.surface_type = 1.0
    simulation.setup()
    simulation.run()
    y = np.copy(mwi.y)

    atmosphere_r = Atmosphere1D(absorbers = [O2(), N2(), H2O()],
                                surface = surface_2)
    simulation_r = ArtsSimulation(atmosphere = atmosphere_r,
                                  data_provider = data_provider,
                                  sensors = [mwi])
    simulation_r.setup()
    simulation_r.run()
    y_r = np.copy(mwi.y)

    assert(np.allclose(y, y_r))
