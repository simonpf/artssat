"""
data_provider
-------------

The :code:`data_provider` module provides a base class for data provider
classes. This class provides utility functions for the composition of data
 providers as well as the overriding of get functions.
"""
from pathlib import Path
import weakref

import numpy as np
import xarray as xr

################################################################################
# Data provider classes
################################################################################


class DataProviderBase:
    """
    The :code:`DataProviderBase` implements generic functionality for

    - The overriding of get methods with attributes
    - Composition of data providers

    When a data provider inherits from :code:`DataProviderBase`, the values
    returned from a :code:`get_<attribute>` can be overridden simply setting
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
        """The super provided."""
        owner = self._owner()
        if owner:
            return owner
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

        attribute_name = name[4:]
        try:
            attr = object.__getattribute__(self, attribute_name)

            def wrapper(*_, **__):
                return attr

            return wrapper
        except AttributeError:
            pass

        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            pass

        for prov in self.subproviders:
            try:
                return getattr(prov, name)
            except AttributeError:
                pass

        raise AttributeError(f"'{self}' object has no attribute '{name}'.")


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
        if not isinstance(provider, DataProviderBase):
            raise Exception(
                "Data provider to add  objects must inherit from DataProviderBase."
            )
        provider.owner = self
        self.providers += [provider]

    def __getattribute__(self, name):

        if not name[:4] == "get_":
            return object.__getattribute__(self, name)

        for prov in self.providers:
            try:
                return getattr(prov, name)
            except AttributeError:
                pass

        raise AttributeError(f"'{self}' object has no attribute 'name'.")


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

    def get(self, *_, **__):
        """Template function for the final get_'name' method."""
        return self.value


class FunctorDataProvider(DataProviderBase):
    """
    The FunctorDataProvider turns a function into a data provider that
    provides a get function for the results of the given function applied
    to a variable from the parent provider.
    """

    def __init__(self, name, variable, func):
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

            func(:code:`function`): Function to apply to the values of
                :code:`variable`. The results are returned as the values
                of the quantity :code:`name` by the :code:`FunctionDataProvider`
                object.
        """
        super().__init__()
        self.variable = variable
        self.func = func
        self.__dict__["get_" + name] = self.get

    def get(self, *args, **kwargs):
        try:
            f_get = getattr(self.owner, "get_" + self.variable)
            f_in = f_get(*args, **kwargs)
        except AttributeError:
            raise Exception(
                f"Could not get variable {self.variable} from data provider."
            )

        result = self.func(f_in)
        return result


class Fascod(DataProviderBase):
    """
    Data provider for 1D Fascod atmospheres.
    """

    def __init__(
        self, climate="midlatitude", season="summer", altitudes=None, pressures=None
    ):
        """
        Args:
            climate: The climate for which to load the atmosphere data. Should
                be one of ['tropical', 'midlatitude', 'subarctic']
            season: The season for which to load the atmosphere data. Should
                be one of ['summer', 'winter'].
            altitudes: If provided, data will be provided on the gives altitude
                grid.
            pressures: If provided, data will be provided on the given pressure
                grid.
        """
        super().__init__()

        climate = climate.lower()
        season = season.lower()

        climates = ["midlatitude", "tropical", "subarctic"]
        if climate not in climates:
            raise ValueError(
                f"The climate for the Fascod data provider must be one of {climates}."
            )
        seasons = ["summer", "winter"]
        if season not in seasons:
            raise ValueError(
                "The season for the Fascod data provider must be one of {seasons}."
            )

        data_folder = Path(__file__).parent / "files" / "fascod"
        if climate == "tropical":
            filename = f"{climate}.nc"
        else:
            filename = f"{climate}_{season}.nc"

        self.data_raw = xr.load_dataset(data_folder / filename)
        self.data = self.data_raw

        if altitudes is not None:
            self.interpolate_altitude(altitudes)
        elif pressures is not None:
            self.interpolate_pressure(pressures)

    def interpolate_altitude(self, altitudes, extrapolate=False, method="linear"):
        """
        Interpolate data in altitude.

        Interpolates the data of the provider to a given altitude grid. After that
        all get methods will return data on the given grid.

        Args:
            altitudes: The altitudes to interpolate the data to.

        """
        kwargs = {"fill_value": np.nan}
        if extrapolate:
            kwargs["fill_value"] = "extrapolate"
        self.data = self.data.interp(z=altitudes, method=method, kwargs=kwargs)

    def interpolate_pressure(self, pressures, extrapolate=False, method="linear"):
        """
        Interpolate data in pressure.

        Interpolates the data of the provider to a given pressure grid. After that
        all get methods will return data on the given grid.

        Args:
            pressures:

        """
        kwargs = {"fill_value": np.nan}
        if extrapolate:
            kwargs["fill_value"] = "extrapolate"

        log_pressures = np.log(pressures)
        data = self.data_raw.copy()
        log_p = np.log(data.p.data)

        data.coords["log_p"] = (("z",), log_p)
        data = data.swap_dims({"z": "log_p"})[{"log_p": slice(None, None, -1)}]

        data = data.interp(log_p=log_pressures, method=method, kwargs=kwargs)
        self.data = data

    def get_temperature(self, *args, **kwargs):
        """Return temperature in atmospheric column."""
        return self.data.t.data

    def get_pressure(self, *args, **kwargs):
        """Return pressure in atmospheric column."""
        return self.data.p.data

    def get_altitude(self, *args, **kwargs):
        """Return altitude in atmospheric column."""
        return self.data.z.data

    def get_O2(self, *args, **kwargs):
        """Return O2 VMR in atmospheric column."""
        return self.data.O2.data

    def get_H2O(self, *args, **kwargs):
        """Return water vapor VMR in atmospheric column."""
        return self.data.H2O.data

    def get_CO2(self, *args, **kwargs):
        """Return CO2 VMR in atmospheric column."""
        return self.data.H2O.data

    def get_O3(self, *args, **kwargs):
        """Return CO3 VMR in atmospheric column."""
        return self.data.H2O.data

    def get_CO(self, *args, **kwargs):
        """Return CO VMR in atmospheric column."""
        return self.data.H2O.data

    def get_N2O(self, *args, **kwargs):
        """Return N2O VMR in atmospheric column."""
        return self.data.H2O.data

    def get_N2(self, *args, **kwargs):
        """Return N2 VMR in atmospheric column."""
        return self.data.H2O.data

    def get_NO2(self, *args, **kwargs):
        """Return NO2 VMR in atmospheric column."""
        return self.data.H2O.data

    def get_NO(self, *args, **kwargs):
        """Return NO VMR in atmospheric column."""
        return self.data.H2O.data

    def get_surface_temperature(self, *args, **kargs):
        """Returns temperature at lowest level."""
        return self.data.t.data[0]
