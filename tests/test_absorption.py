"""
Tests for setting up the absorption in an artssat simulation.
"""
import numpy as np

from artssat import ArtsSimulation
from artssat.atmosphere import Atmosphere1D
from artssat.sensor import CloudSat, ICI
from artssat.atmosphere.absorption import O2
from artssat.atmosphere.surface import Tessem
from artssat.data_provider import Fascod


def test_absorption_o2():
    atmosphere = Atmosphere1D(absorbers=[O2()], surface=Tessem())
    ici = ICI()
    ici.sensor_line_of_sight = np.array([[135.0]])
    ici.sensor_position = np.array([[600e3]])

    simulation = ArtsSimulation(atmosphere = atmosphere,
                                data_provider = Fascod,
                                sensors = [ici])
    simulation.setup()
    assert simulation.workspace.abs_species.value[0][0][:8] == "O2-PWR98"
