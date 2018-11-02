"""
data_provider
-------------

The :code:`data_provider` module provides a base class for data provider
classes. This class provides utility function for the composition of data providers
as well as the overriding of get functions.
"""

import numpy as np

################################################################################
# Data provider classes
################################################################################

class DataProviderBase:
    """
    The :code:`DataProviderBase` implements generic functionality for

    - The overriding of get methods with attributes
    - Composition of data providers

    When a data provider inherits from :code:`DataProviderBase`, the values
    returned from a :code:`get_<attribute>` can be overriding simply setting
    the attribute with the name :code:`<attribute>` of the object.

    The DataProviderBase also allows composing data provider by adding sub-
    providers to a data provider object. If a simulation queries a get_method
    from the data provider that the data provider doesn't provide, the query
    is forwarded to the subproviders in the order they were added. The get
    method from the first that provides such a method is returned.
    """
    def __init__(self):
        self.owner = None
        self.subproviders = []

    def add(self, subprovider):
        """
        Add a subprovider to the data provider.

        Arguments:

            subprovider: The subprovider to add. Must inherit
                from :code:`DataProviderBase`.

        Raises:

            Exception: If :code: `subprovider` argument doesn´t inherit
            from :code:`DataProviderBase`.
        """
        if not isinstance(subprovider, DataProviderBase):
            raise Exception("Subprovider objects must inherit from DataProviderBase.")
        subprovider.owner = self
        self.subproviders += [subprovider]

    def __getattribute__(self, name):

        if not name[:4] == "get_":
            return object.__getattribute__(self, name)
        else:

            attribute_name = name[4:]

            #
            # Get name from self with priority for the attribute.
            #

            try:
                x = object.__getattribute__(self, attribute_name)

                def wrapper(*args, **kwargs):
                    return x
                return wrapper
            except:
                pass

            try:
                return object.__getattribute__(self, name)
            except:
                pass

            #
            # Get name from first list provider which has any of the
            # has either a fitting attribute or get_method.
            #

            for p in self.subproviders:

                try:
                    return getattr(p, name)
                except:
                    pass

            raise AttributeError("'{0}' object has no attribute '{1}'."\
                                 .format(type(self).__name__, name))

################################################################################
# A priori providers
################################################################################

class FixedApriori(DataProviderBase):
    """
    A priori provider for fixed a priori mean and covariance.

    The :code:`FixedApriori` class provides get methods for a fixed a priori
    mean and covariance matrix. It is intended to be used as a subprovider
    to a main data provider.
    """

    def __init__(self,
                 name,
                 xa,
                 covariance,
                 xr = 1e-12,
                 height_mask = None,
                 temperature_mask = None,
                 spatial_correlation = None):
        """
        Create a priori provider for quantity :code:`name`. The created
        object will provide :code:`get_<name>_xa` and
        :code:`get_<name>_covariance` methods that return fixed a priori vectors
        :math:`x_a` and diagonal covariance matrices with the values given
        by :code:`xa` and :code:`covariance`, respectively.

        Arguments:

            name(code:`str`): The name of the retrieval quantity for which the
                a priori should be provided.

            xa(code:`float` or :code:`np.ndarray`): If a :code:`float` is given,
                the a priori is assumed constant throughout the atmosphere. If a
                vector is given this one is returned as the a priori profile.

            covariance(code:`float` or :code:`np.ndarray`): Diagonal values
                of the covariance matrix to be returned. If only a single value
                is given it is extended to the whole atmosphere.

            height_mask(callable): Callable object that applies a height mask
                to the a priori vector and the covariance matrix.

            temperature_mask(callable): Same as height mask but for the
                temperature profile of the atmosphere.

            spatial_correlation: Callable that applies spatial correlation to
                covariance matrix.
        """

        super().__init__()

        xa_name = "get_" + name + "_xa"
        self.__dict__[xa_name] = self.get_xa
        covariance_name = "get_" + name + "_covariance"
        self.__dict__[covariance_name] = self.get_covariance

        self.xa  = xa
        self.xr = xr
        self.covariance = covariance
        self.height_mask = height_mask
        self.temperature_mask = temperature_mask
        self.spatial_correlation = spatial_correlation

    def get_xa(self, *args, **kwargs):
        """
        Function to which the :code:`get_<name>_xa` call is forwarded.
        """

        z = self.owner.get_altitude(*args, **kwargs)
        t = self.owner.get_temperature(*args, **kwargs)
        x = self.xa * np.ones(z.shape)

        if not self.height_mask is None:
            x = self.height_mask(x, self.xr, z)

        if not self.temperature_mask is None:
            x = self.temperature_mask(x, self.xr, t)

        return x

    def get_covariance(self, *args, **kwargs):
        """
        Function to which the :code:`get_<name>_covariance` call is forwarded.
        """

        z = self.owner.get_altitude(*args, **kwargs)
        t = self.owner.get_temperature(*args, **kwargs)
        covmat = np.diag(self.covariance * np.ones(z.size))

        if not self.spatial_correlation is None:
            covmat = self.spatial_correlation(covmat, z)

        if not self.height_mask is None:
            covmat = self.height_mask.apply_matrix(covmat, 1e-12, z)

        if not self.temperature_mask is None:
            covmat = self.temperature_mask.apply_matrix(covmat, 1e-12, t)

        return covmat
