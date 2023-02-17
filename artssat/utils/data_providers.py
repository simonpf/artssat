"""
artssat.utils.data_providers
----------------------------

The :code:`utils.data_providers` module provides commodity functions
and class that simplify access and combination of input data for
artssat retrievals.
"""

from artssat.data_provider import DataProviderBase
from netCDF4 import Dataset
import numpy as np


class NetCDFDataProvider(DataProviderBase):
    """
    The NetCDFDataProvider class exposes the variables in a
    NetCDF file using the :code:`artssat` data interface. For
    each variable :code`<varname>` in the NetCDF file, the
    resulting :code:`NetCDFDataProvider` object has a get
    method :code:`get_<varname>(*args)` for every variable
    in the NetCDF file.

    The positional arguments :code:`*args` given to the
    get methods are consecutively applied to the variable.
    """

    @staticmethod
    def _make_getter(variable, fixed_dimensions, offsets):
        def get(*args):

            args = list(args)
            args.reverse()
            v = variable

            i = 0
            if not len(args):
                v = v[:]
            else:
                while len(args) > 0:
                    d = variable.dimensions[i]
                    i += 1
                    if d in fixed_dimensions:
                        v = v[fixed_dimensions[d]]
                    else:
                        a = args.pop()
                        if d in offsets:
                            a += offsets[d]
                        v = v[a]

            if type(v) == np.ma.core.MaskedArray:
                v = np.ma.getdata(v)

            if len(v.shape) > 0:
                return v[:]
            else:
                return v

        return get

    def __init__(self, path, *args, group=None, **kwargs):
        """
        Arguments:

            path(str): Path to a NetCDF4 file.

        """
        self.file_handle = Dataset(path, *args, **kwargs)
        self.fixed_dimensions = {}
        self.offsets = {}

        if group is None:
            group = self.file_handle
        elif type(group) == int:
            k = list(self.file_handle.groups.keys())[group]
            group = self.file_handle.groups[k]
        else:
            group = self.file_handle.groups[group]

        for name in group.variables:
            fname = "get_" + name
            v = group.variables[name]
            self.__dict__[fname] = NetCDFDataProvider._make_getter(
                v, self.fixed_dimensions, self.offsets
            )

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

    def add_offset(self, dimension, value):
        """
        This adds an offset to a dimension. The value is then added to
        any arguments provided to any get method along that dimension.

        Arguments:

            dimension(str): Name of the dimension that is offset.

            value(int): The value that should be added to the argument.
        """
        self.offsets[dimension] = value
