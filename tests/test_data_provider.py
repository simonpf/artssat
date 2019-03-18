from parts.data_provider import DataProviderBase, Constant, \
    FunctorDataProvider, CombinedProvider

################################################################################
# Data provider classes
################################################################################

class DataProvider(DataProviderBase):

    def __init__(self):
        super().__init__()
        self.temperature = 275.0

    def get_temperature(self):
        return 300.0

class SubProvider1(DataProviderBase):

    def __init__(self):
        super().__init__()

    def get_temperature_xa(self):
        return 300.0

class SubProvider2(DataProviderBase):

    def __init__(self):
        super().__init__()
        self.temperature_xa =  300.0

################################################################################
# Tests
################################################################################

def test_priority():
    """
    Test that attributes take priority over get functions.

    We first test the get function which should return the value of the
    temperature attribute instead of calling the get_temperature method.

    Then we remove the temperature attribute and check that the returned
    value is the one returned by the get_temperature method.
    """
    dp  = DataProvider()
    assert(dp.get_temperature() == 275.0)

    dp.__dict__.pop("temperature")
    assert(dp.get_temperature() == 300.0)

def test_subproviders():
    """
    Test that call to get_temperature_xa is correctly forwarded to the
    get_temperature temperature method of the Subprovider1 object, which
    comes first in the list of data providers.
    """

    dp  = DataProvider()
    sp1 = SubProvider1()
    sp2 = SubProvider2()

    dp.add(sp1)
    dp.add(sp2)

    assert(dp.get_temperature_xa() == 300.0)

def test_combined_provider():
    """
    Test priority with which getters are returned from the
    CombinedDataProvider.
    """
    dp_1 = DataProvider()
    dp_1.add(FunctorDataProvider("value", "temperature", lambda x: 2.0 * x))

    dp_2 = DataProvider()
    dp_2.add(FunctorDataProvider("value", "temperature", lambda x: 3.0 * x))

    dp = CombinedProvider(dp_1, dp_2)
    assert(dp.get_value() == 2.0 * dp.get_temperature())

    dp = CombinedProvider(dp_2, dp_1)
    assert(dp.get_value() == 3.0 * dp.get_temperature())

def test_constant_provider():
    dp = Constant("value", 100)
    value = dp.get_value()
    assert(value == 100)

def test_functional_provider():
    dp = DataProvider()
    dp.add(FunctorDataProvider("value", "temperature", lambda x: 2.0 * x))
    value = dp.get_value()
    assert(value == 2.0 * dp.get_temperature())

