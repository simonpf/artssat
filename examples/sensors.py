from parts.sensor import PassiveSensor
import numpy as np

class ICI(PassiveSensor):
    """
    The Ice Cloud Imager (ICI) sensor.

    Attributes:

        channels(:code:`list`): List of channels that are available
            from ICI

        nedt(:code:`list`): Noise equivalent temperature differences for the
            channels in :code:`channels`.
    """
    channels = np.array([1.749100000000000e+11,
                         1.799100000000000e+11,
                         1.813100000000000e+11,
                         2.407000000000000e+11,
                         3.156500000000000e+11,
                         3.216500000000000e+11,
                         3.236500000000000e+11,
                         4.408000000000000e+11,
                         4.450000000000000e+11,
                         4.466000000000000e+11,
                         6.598000000000000e+11])

    nedt = np.array([0.8, 0.8, 0.8,       # 183 GHz
                     0.7 * np.sqrt(0.5),  # 243 GHz
                     1.2, 1.3, 1.5,       # 325 GHz
                     1.4, 1.6, 2.0,       # 448 GHz
                     1.6 * np.sqrt(0.5)]) # 664 GHz

    def __init__(self,
                 name = "ici",
                 channel_indices = None,
                 stokes_dimension = 1):
        """
        This creates an instance of the ICI sensor to be used within a
        :code:`parts` simulation.

        Arguments:

            name(:code:`str`): The name of the sensor used within the parts
                simulation.

            channel_indices(:code:`list`): List of channel indices to be used
                in the simulation/retrieval.

            stokes_dimension(:code:`int`): The stokes dimension to use for
                the retrievals.
        """
        if channel_indices is None:
            channels  = ICI.channels
            self.nedt = ICI.nedt
        else:
            channels  = ICI.channels[channel_indices]
            self.nedt = self.nedt[channel_indices]
        super().__init__(name, channels, stokes_dimension = stokes_dimension)
