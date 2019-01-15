"""
parts.utils.data_providers
--------------------------

The :code:`utils.data_providers` module provides commodity functions
and class that simplify access and combination of input data for
parts retrievals.
"""

from parts.data_provider import DataProviderBase
from netCDF4 import Dataset

class NetCDFDataProvider(DataProviderBase):
    """
    The NetCDFDataProvider class exposes the variables in a
    NetCDF file using the :code:`parts` data interface. For
    each variable :code`<varname>` in the NetCDF file, the
    resulting :code:`NetCDFDataProvider` object has a get
    method :code:`get_<varname>(*args)` for every variable
    in the NetCDF file.

    The positional arguments :code:`*args` given to the
    get methods are consecutively applied to the variable.
    """
    @staticmethod
    def _make_getter(variable):
        def get(*args):
            v = variable
            for a in args:
                v = variable[a]
            if len(v.shape) > 0:
                return v[:]
            else:
                return v
        return get



    def __init__(self, path):
        """
        Arguments:

            path(str): Path to a NetCDF4 file.

        """
        self.file_handle = Dataset(path)

        for name in self.file_handle.variables:
            fname = "get_" + name
            v = self.file_handle.variables[name]
            self.__dict__[fname] = NetCDFDataProvider._make_getter(v)

        super().__init__()
