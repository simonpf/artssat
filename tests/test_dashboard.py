#ip = get_ipython()
#ip.magic("load_ext autoreload")
#ip.magic("autoreload 2")


import numpy as np
from examples.data_provider import DataProvider

import parts
from parts                       import ArtsSimulation
from parts.atmosphere            import Atmosphere1D
from parts.atmosphere.absorption import O2, N2, H2O
from parts.atmosphere.surface    import Tessem
from parts.jacobian              import Log10
from parts.scattering            import ScatteringSpecies, D14
from parts.scattering.solvers    import RT4, Disort
from parts.sensor                import CloudSat, ICI
from parts.data_provider         import DataProviderBase
from parts.retrieval.a_priori    import FixedAPriori
from parts.dashboard             import dashboard

from examples.data_provider      import DataProvider

import os
import sys
test_path = os.path.join(os.path.dirname(parts.__file__), "..", "tests")
sys.path.append(test_path)
from utils.data import scattering_data, scattering_meta

import matplotlib.pyplot as plt

#from IPython import get_ipython

#
# Scatterers
#

ice = ScatteringSpecies("ice", D14(-1.0, 2.0),
                        scattering_data = scattering_data,
                        scattering_meta_data = scattering_meta)

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

sensors = [cs, ici]


atmosphere = Atmosphere1D(absorbers = [O2(), N2(), H2O()],
                          scatterers = [ice],
                          surface = Tessem())
simulation = ArtsSimulation(atmosphere = atmosphere,
                            sensors = sensors)
data_provider = DataProvider()
simulation.data_provider = data_provider
simulation.setup()
simulation.run()

from parts.dashboard import standalone_server
