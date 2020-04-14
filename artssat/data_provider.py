"""
data_provider
-------------

The :code:`data_provider` module provides a base class for data provider
classes. This class provides utility function for the composition of data providers
as well as the overriding of get functions.
"""

import numpy as np
import weakref

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
        self._owner = None
        self.subproviders = []

    @property
    def owner(self):
        owner = self._owner()
        if owner:
            return owner
        else:
            raise ValueError("Parent data provider has been deleted.")

    @owner.setter
    def owner(self, owner):
        self._owner = weakref.ref(owner)

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

class CombinedProvider(DataProviderBase):
    """
    The :code:`CombinedProvider` allows the combination of multiple data
    providers. If a get method is requested from the data provider it is
    forwarded to the combined providers. The lookup is performed in the
    same order as the providers are provided to the constructor and the
    first hit is returned.
    """
    def __init__(self, *args):
        super().__init__()
        self.providers = list(args)

    def add(self, provider):
        """
        Add a provider to the combined data provider.

        Arguments:

            provider: The data provider to add. Must inherit
                from :code:`DataProviderBase`.

        Raises:

            Exception: If :code: `provider` argument doesn´t inherit
            from :code:`DataProviderBase`.
        """
        if not isinstance(subprovider, DataProviderBase):
            raise Exception("Data provider to add  objects must inherit from "
                            "DataProviderBase.")
        subprovider.owner = self
        self.providers += [provider]

    def __getattribute__(self, name):

        if not name[:4] == "get_":
            return object.__getattribute__(self, name)
        else:

            #
            # Get name from first list provider which has any of the
            # has either a fitting attribute or get_method.
            #

            for p in self.providers:

                try:
                    return getattr(p, name)
                except:
                    pass

            raise AttributeError("'{0}' object has no attribute '{1}'."\
                                 .format(type(self).__name__, name))

class Constant(DataProviderBase):
    """
    Returns a data provider that returns a constant value for a given
    quantity.
    """
    def __init__(self, name, value):
        """
        Create a data provider for quantity :code:`name` that
        returns the fixed value :code:`value`.

        Arguments:

            name(:code:`str`): Name of the quantity to provide. The
                created :class:`FunctorDataProvider` object will have
                a member function name :code:`get_<name>`.

            value: The value to return for the quantity :code:`quantity`.
        """
        super().__init__()
        self.value = value
        self.__dict__["get_" + name] = self.get

    def get(self, *args, **kwargs):
        return self.value

class FunctorDataProvider(DataProviderBase):
    """
    The FunctorDataProvider turns a function into a data provider that
    provides a get function for the results of the given function applied
    to a variable from the parent provider.
    """
    def __init__(self, name, variable, f):
        """
        Create a data provider for quantity :code:`name` that
        returns the result of function `f` applied to variable
        `name` from the parent provider.

        Arguments:

            name(:code:`str`): Name of the quantity to provide. The
                created :class:`FunctorDataProvider` object will have
                a member function name :code:`get_<name>`.

            variable(:code:`str`): The quantity to get from the parent
                data provider. When its get function is called the
                :code:`FunctorDataProvider` will call the :code:`get_<variable>`
                function of the parent data provider and the values as
                arguments to the function :code:`f`

            f(:code:`function`): Function to apply to the values of
                :code:`variable`. The results are returned as the values
                of the quantity :code:`name` by the :code:`FunctionDataProvider`
                object.
        """
        super().__init__()
        self.variable   = variable
        self.f          = f
        self.__dict__["get_" + name] = self.get

    def get(self, *args, **kwargs):
        try:
            f_get = getattr(self.owner, "get_" + self.variable)
            x = f_get(*args, **kwargs)
        except:
            raise Exception("Could not get variable {} from data provider."
                            .format(self.variable))

        y = self.f(x)
        return y
