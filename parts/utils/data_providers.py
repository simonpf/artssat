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
    def _make_getter(variable, fixed_dimensions):
        def get(*args):

            args = list(args)
            args.reverse()
            v = variable

            i = 0
            while len(args) > 0:
                d = variable.dimensions[i]
                if d in fixed_dimensions:
                    v = v[fixed_dimensions[d]]
                    i += 1
                else:
                    v = v[args.pop()]

            if len(v.shape) > 0:
                return v[:]
            else:
                return v
        return get

    def __init__(self, path, *args, **kwargs):
        """
        Arguments:

            path(str): Path to a NetCDF4 file.

        """
        self.file_handle      = Dataset(path, *args, **kwargs)
        self.fixed_dimensions = {}

        for name in self.file_handle.variables:
            fname = "get_" + name
            v = self.file_handle.variables[name]
            self.__dict__[fname] = NetCDFDataProvider._make_getter(v, self.fixed_dimensions)

        super().__init__()

    def fix_dimension(self, dimension, value):
        """
        Fixed the value of the dimension :code:`dimension` for every variables in
        the NetCDF file. This can be used if the data provider is combined with
        another one that has less dimensions.

        Arguments:

            dimension(str): Name of the dimension to fix.

            value(int): The value that the dimension should be fixed to.
        """
        self.fixed_dimensions[dimension] = value
