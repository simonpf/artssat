"""
PSD data
========

Particle size distribution (PSD) data consists of a range of particle
densities given w.r.t to a corresponding grid of particle sizes. This
module prvodies functionality for the handling of such PSD data.

Class reference
---------------
"""
import numpy as np

################################################################################
# Size parameter classes
################################################################################

class SizeParameter:
    """
    General representation of a size parameter of a PSD. The size parameter
    is represented by its mass-size relation, which gives the mass :math:`m`
    as a function of the size parameter :math:`X`:

    .. math::
        m(X) = a \cdot X^b

    Attributes:

        a(numpy.float): The :math:`a` factor of the mass-size relation

        b(numpy.float): The :math:`b` exponent of the mass-size-relation
    """

    def __eq__(self, other):
        return self.a == other.a and self.b == other.b

    def __init__(self, a, b):
        """
        Create size parameter with given :math:`a` and :math:`b` parameters.
        """
        self.a = a
        self.b = b

    def convert(self, src, x, y):
        """
        Conversion of PSD data given over one size parameter to another.

        Parameters:

            src(SizeParameter): The size parameter in which the PSD data is given.

            x(numpy.ndarray): The size grid over which the PSD data is given.
            Must be broadcastable into the shape of :code:`y`.

            y(numpy.ndarray): The PSD data with the last axis corresponding to
            the size grid :code:`x`.

        Returns:

            :code:`(x, y)`: Tuple containing the transformed size grid :code:`x`
            and PSD data :code:`y`
        """
        x = (src.a / self.a) ** (1.0 / self.b) * x ** (src.b / self.b)
        y = y * (self.a / src.a) ** (1.0 / src.b) * (self.b / src.b) \
            * x ** (self.b / src.b - 1.0)
        return x, y

    def get_mass_density(self, x, y):
        """
        Compute the mass of the PSD for given PSD data.

        Parameters:

            x(numpy.ndarray): The size grid over which the PSD data is given.
            Must be broadcastable into the shape of :code:`y`.

            y(numpy.ndarray): The PSD data with the last axis corresponding to
            the size grid :code:`x`.

        """
        return np.trapz(self.a * x ** self.b * y, x = x, axis = -1)

class Area(SizeParameter):
    def __ini__(self, a, b):
        super().__init__(a, b)

class D_eq(SizeParameter):

    def __init__(self, rho):
        self.rho = rho
        super().__init__(self.rho * np.pi / 6.0, 3.0)

class D_max(SizeParameter):
    def __init__(self, a, b):
        super().__init__(a, b)

class Mass(SizeParameter):
    def __init__(self):
        super().__init__(1.0, 1.0)

################################################################################
# PSD Data
################################################################################

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
    def __init__(self, x, data, size_parameter):
        """
        Create a PSDData object from data.

        The particle number density values should be given as a 2D array
        :code:`data`, with the first dimenions corresponding to PSDs from
        different volume elements and the first dimension corresponding to
        the size paramter.

        Parameters:

            x(numpy.array): The size grid used for :code:`data`

            data(numpy.array): Array containing particle number density values

            size_paramter(SizeParameter): Enum representing the type of size
                parameter used.

        """
        x_shape = (len(data.shape) - 1) * (1,) + (-1,)
        x = x.reshape(x_shape)

        if data.shape[-1] != x.shape[-1]:
            raise Exception("Size grid 'x' and PSD data 'y' must have the same"\
                            " number of elements along the last dimension.")


        self.data = data
        self.x = x

        if not isinstance(size_parameter, SizeParameter):
            raise Exception("Provided size_parameter is not a valid value of "
                            "of type SizeParameter.")
        self.size_parameter = size_parameter

    def get_moment(self, p, reference_size_parameter = None):
        """
        Compute the :math:`p` th moment :math:M(p) of the PSD data:

        .. math::
            M(p) = \int_0^\infty X^P \frac{dN}{dX}(X)\:dX

        The physical significance of a moment of a PSD depends on the size
        parameter. So in general, the moments of the same PSD given w.r.t.
        different size parameters differ. If :code: `reference_size_parameter`
        argument is given then the computed moment will correspond to the
        Moment of the PSD w.r.t. to the given size parameter.

        Parameters:

            p(numpy.float): The moment which to compute.

            reference_size_parameter(:class: `SizeParameter`): Size parameter
            with respect to which the moment should be computed.

        Return:
            The :math:`p` th moment of the particle size distribution.

        """
        if not reference_size_parameter is None:
            a1 = self.size_parameter.a
            b1 = self.size_parameter.b
            a2 = reference_size_parameter.a
            b2 = reference_size_parameter.b

            c = (a1 / a2) ** (p / b2)
            p = p * b1 / b2
        else:
            c = 1.0

        return c * np.trapz(self.data * self.x ** p, x = self.x)

    def __add__(self, other):
        """
        Adds the data from two particle size distributions. For this the PSDs
        must be given w.r.t to the same size parameter type and grid.

        Parameters:

            other(:code:`PSDData`): The PSD to add.

        Returns:

            :code:`PSDData` object representing the sum of the two PSDs.

        """
        # TODO: Make an abstract base class for PSDs
        if hasattr(other, "evaluate"):
            x2, _ = other.size_parameter.convert(self.size_parameter,
                                                 self.x,
                                                 self.data)
            other = other.evaluate(self.x2)
            other.change_size_parameter(self.size_parameter)

        if (not self.size_parameter.a == other.size_parameter.a) or \
           (not self.size_parameter.b == other.size_parameter.b):
           raise Exception("Addition of PSD data is only defined for PSDs"
                           " given w.r.t. the same size parameter.")

        if not np.all(np.isclose(self.x, other.x)):
            raise Exception("Addition of PSD data is only defined for PSDs"
                            " defined over the same size grid.")

        size_parameter = self.size_parameter
        y = self.data + other.data
        x = self.x

        return PSDData(x, y, size_parameter)

    def get_number_density(self):
        """
        Computes the total particle number density (:math:`0` th moment).

        Returns:
            The particle number density for each volume element.

        """
        return self.get_moment(p = 0)

    def get_mass_density(self):
        """
        Computes the particle mass density corresponding to the PSD data.

        Returns:
            Array containing the particle mass density for each of the
            bulk volumes described by the PSD data
        """
        return self.size_parameter.get_mass_density(self.x, self.data)

    def change_size_parameter(self, size_parameter):
        """
        Change the size parameter of the data.

        Parameters:
            size_parameter(SizeParameter): The size parameter to convert the
                data to.

            rho(np.float): Particle density.

        """

        self.x, self.data = size_parameter.convert(self.size_parameter,
                                                   self.x, self.data)
        self.size_parameter = size_parameter
