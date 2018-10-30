"""
data_provider
=============

The :code:`data_provider` module provides a base class for data provider
classes. This class provides utility function for the composition of data providers
as well as the overriding of get functions.
"""

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
