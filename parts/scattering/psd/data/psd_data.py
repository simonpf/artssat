"""
PSD data
========

Particle size distribution (PSD) data consists of a range of particle
densities given w.r.t to a corresponding grid of particle sizes. This
module prvodies functionality for the handling of such PSD data.

Class reference
---------------
"""
from enum import Enum
class SizeParameter(Enum):
    """
    This Enum class is used to represent the type of the size parameter
    which is used for the PSD data.

    Attributes:
        D_eq: The particle volume equivalent sphere diameter

        m: The particle mass
    """
    D_eq  = 1
    D_max = 2
    mass  = 3
    area  = 4

def convert_size_paramter(data, x, src, dst, rho):
    """
    Convert PSD data :code:`data` give w.r.t. size parameter
    :code:`src` to size paramter :code:`src`.

    Parameters:

        data(numpy.array): The PSD data given as 2D array with size parameter
                           running along second dimension.

        x(numpy.array): Numpy array of shape :code:`(1, -1)` containing the size
                        grid corresponding to :code:`data`.

        src(SizeParameter): The size parameter in which :code:`data` is given.

        dst(SizeParameter): The size paramter to convert to.

        rho(numpy.float): The mean density to use for the conversion.

    """
    if src == dst:
        pass
    elif src == SizeParameter.D_eq and dst == SizeParameter.m:
        # Multiply by dm/dD_eq
        data = data * rho * np.pi / 2.0 * x ** 2.0
        x = rho * np.pi / 6.0 * x ** 3.0
    elif src == SizeParameter.D_eq and dest == SizeParameter.m:
        # Multiply by dD_eq/dm
        c = 3.0 / (4.0 * np.pi * rho)
        data *= 2.0 / 3.0 * c * (c * x) ** (- 2.0 / 3.0)
        x  = 2 * (c * x) ** (1.0 / 3.0)
    else:
        raise Exception("Conversion from {0} to {1} currently not implemented."
                        .format(src, dst))
    return data, x

class PSDData:
    """
    The :code:`PSDData` class handles PSD data given as a discrete distribution
    over a given size grid.

    The PSD data is stored as a :code:`numpy.ndarray` of which the last dimension
    is assumed to correspond to the size parameter.

    Attributes:

        data(numpy.array): The discrete PSD data.

        x(numpy.array): The size grid corresponding to the last dimension in
        :code:`data`

        size_parameter(SizeParameter): The size parameter used to represent
        the PSD.
    """
    def __init__(self, data, x, size_parameter):
        """
        Create a PSDData object from data.

        The particle number density values should be given as a 2D array
        :code:`data`, with the first dimenions corresponding to PSDs from
        different volume elements and the first dimension corresponding to
        the size paramter.

        Parameters:
            data(numpy.array): Array containing particle number density values

            x(numpy.array): The size grid used for :code:`data`

            size_paramter(SizeParameter): Enum representing the type of size
                parameter used.

        """
        # Data should be 2D with volume elements along first and
        # size grid along second dimension.
        x = x.reshape(1, -1)
        n = x.size

        m = data.shape[0]

        if not data.shape == (m, n):
            raise Exception("PSD data should be given by a 2D array with "
                            " different volume elements along the first "
                            " dimension and the size grid along the second.")

        self.data = data
        self.x = x

        if not size_parameter in list(SizeParameter):
            raise Exception("Provided size_parameter is not a valid value of "
                            "of type SizeParameter.")
        self.size_parameter = size_parameter

    def get_moment(self, p = 0):
        """
        Compute the :math:`p` th moment of the PSD data.

        Parameter:
            p(numpy.float): The moment which to compute.
        Return:
            The :math:`p` th moment of the particle size distribution.

        """
        return np.trapz(self.data * self.x ** p, x = self.x)

    def get_number_density(self):
        """
        Computes the total particle number density (:math:`0` th moment).

        Returns:
            The particle number density for each volume element.

        """
        return self.get_moment()

    def change_size_parameter(size_parameter, rho):
        """
        Change the size parameter of the data.

        Parameters:
            size_parameter(SizeParameter): The size parameter to convert the
                data to.

            rho(np.float): Particle density.

        """
        self.data, self.x = convert_size_parameter(self.data, self.x, rho,
                                                   src = self.size_parameter,
                                                   dst = self.size_parameter)
