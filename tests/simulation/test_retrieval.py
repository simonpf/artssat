import numpy as np
import scipy as sp
import pytest

from parts                       import ArtsSimulation
from parts.atmosphere            import Atmosphere1D
from parts.atmosphere.absorption import O2, N2, H2O
from parts.atmosphere.surface    import Tessem
from parts.jacobian              import Log10
from parts.scattering            import ScatteringSpecies, D14
from parts.scattering.solvers    import RT4, Disort
from parts.sensor                import CloudSat, ICI
from parts.data_provider         import DataProviderBase, FixedApriori

from examples.data_provider      import DataProvider
from tests.data                  import scattering_data, scattering_meta

import matplotlib.pyplot as plt

################################################################################
# A priori providers
################################################################################

class IndependentMeasurementErrors(DataProviderBase):
    """
    Provides get method for observation error covariances.
    """
    def __init__(self, n, sigma):
        self.n     = n
        self.sigma = sigma

    def get_observation_error_covariance(self):
        return sp.sparse.diags(self.sigma ** 2 * np.ones(self.n),
                               format = "coo")

################################################################################
# Setup method
################################################################################

def setup_retrieval_simulation(retrieval_type = "passive",
                               scattering = False):
    """
    Setup up a retrieval simulation with or without
    scattering.
    """

    #
    # Scattering
    #

    ice = ScatteringSpecies("ice", D14(-1.0, 2.0),
                            scattering_data = scattering_data,
                            scattering_meta_data = scattering_meta)
    if scattering:
        scatterers = [ice]
    else:
        scatterers = []

    ice.psd.t_min = 0.0
    ice.psd.t_max = 275.0

    #
    # Sensors
    #

    ici = ICI()
    ici.sensor_line_of_sight = np.array([[135.0]])
    ici.sensor_position = np.array([[600e3]])

    cs = CloudSat()
    cs.range_bins = np.linspace(0, 30e3, 31)
    cs.sensor_line_of_sight = np.array([[135.0]])
    cs.sensor_position = np.array([[600e3]])
    cs.y_min = -35.0

    if retrieval_type == "active":
        sensors = [cs]
    elif retrieval_type == "passive":
        sensors = [ici]
    else:
        sensors = [cs, ici]


    atmosphere = Atmosphere1D(absorbers = [O2(), N2(), H2O()],
                              scatterers = scatterers,
                              surface = Tessem())
    simulation = ArtsSimulation(atmosphere = atmosphere,
                                sensors = sensors)
    return simulation

def test_scattering_retrieval_passive():
    simulation = setup_retrieval_simulation(retrieval_type = "passive",
                                            scattering = True)
    data_provider            = DataProvider()
    simulation.data_provider = data_provider
    simulation.setup()
    simulation.run()
    y = np.copy(simulation.workspace.y)

    ice = simulation.atmosphere.scatterers[0]
    simulation.retrieval.add(ice.mass_density)
    ice.mass_density.transformation = Log10()
    ice.mass_density.limit_low = -10
    simulation.retrieval.settings["max_iter"] = 1

    ice_a_priori         = FixedApriori("ice_mass_density", -6, 2.0)
    data_provider.add(ice_a_priori)

    n = simulation.sensors[0].f_grid.size * simulation.sensors[0].stokes_dimension
    measurement_a_priori = IndependentMeasurementErrors(n, 1.0)
    data_provider.add(measurement_a_priori)

    simulation.retrieval.y = y
    simulation.setup()
    simulation.run()
    return simulation.workspace

def test_scattering_retrieval_active():
    simulation = setup_retrieval_simulation(retrieval_type = "active",
                                            scattering = True)
    data_provider = DataProvider()
    simulation.data_provider = data_provider
    simulation.setup()
    simulation.run()
    y = np.copy(simulation.workspace.y)

    ice = simulation.atmosphere.scatterers[0]
    simulation.retrieval.add(ice.mass_density)
    ice.mass_density.limit_low = 1e-6
    simulation.retrieval.settings["max_iter"] = 5
    ice.mass_density.transformation = Log10()

    # Ice a priori
    ice_a_priori         = FixedApriori("ice_mass_density", -6, 2.0)
    data_provider.add(ice_a_priori)

    # Observation error a priori
    n = simulation.sensors[0].range_bins.size - 1
    measurement_a_priori = IndependentMeasurementErrors(n, 1.0)
    data_provider.add(measurement_a_priori)

    simulation.retrieval.y = y
    simulation.setup()
    simulation.run()
    return simulation.workspace

def test_scattering_combined_retrieval():

    simulation = setup_retrieval_simulation(retrieval_type = "combined",
                                            scattering = True)
    data_provider = DataProvider()
    simulation.data_provider = data_provider
    simulation.sensors[1].stokes_dimension = 1
    simulation.setup()

    # Forward simulation
    simulation.run()
    y = np.copy(simulation.workspace.y)

    ice = simulation.atmosphere.scatterers[0]
    simulation.retrieval.add(ice.mass_density)
    simulation.retrieval.add(ice.mass_weighted_diameter)

    ice.mass_density.retrieval.limit_low = -10
    ice.mass_density.transformation = Log10()

    ice.mass_weighted_diameter.retrieval.limit_low = -6
    ice.mass_weighted_diameter.transformation = Log10()


    # Ice a prioris

    ice_a_priori         = FixedApriori("ice_mass_density", -6, 2.0)
    data_provider.add(ice_a_priori)
    ice_a_priori         = FixedApriori("ice_mass_weighted_diameter", -3, 2.0)
    data_provider.add(ice_a_priori)

    # Observation errors

    n  = simulation.sensors[0].range_bins.size - 1
    n += simulation.sensors[1].f_grid.size * simulation.sensors[1].stokes_dimension
    measurement_a_priori = IndependentMeasurementErrors(n, 1.0)
    data_provider.add(measurement_a_priori)

    print(ice.mass_density.transformation)
    simulation.retrieval.settings["max_iter"] = 1
    simulation.retrieval.y = y

    # Retrieval
    simulation.workspace.x = []
    simulation.run()
    y_f = np.copy(simulation.workspace.yf.value)
    x   = np.copy(simulation.workspace.x.value)
    #return x, y_f, y
    return simulation

def test_simulation_absorption_retrieval():

    simulation = setup_retrieval_simulation(retrieval_type = "passive",
                                            scattering = False)
    data_provider = DataProvider()
    simulation.data_provider = data_provider
    simulation.setup()
    simulation.run()

    # O2 a priori
    o2_a_priori         = FixedApriori("O2", 0.209, 0.01)
    data_provider.add(o2_a_priori)

    n = simulation.sensors[0].f_grid.size * simulation.sensors[0].stokes_dimension
    measurement_a_priori = IndependentMeasurementErrors(n, 1.0)
    data_provider.add(measurement_a_priori)

    o2 = simulation.atmosphere.absorbers[0]
    y = np.copy(simulation.workspace.y)
    simulation.retrieval.add(o2)
    simulation.retrieval.y = y

    simulation.setup()
    simulation.run()
