"""
artssat.data.atmosphere
=====================

This module contains standard data providers that provide clear sky
atmospheres for different climatological regimes.
"""
from pathlib import Path

import numpy as np
from artssat.data_provider import DataProviderBase
from pyarts.xml import load


class Tropical(DataProviderBase):
    def __init__(self, p=None, z=None):
        super().__init__()

        self.pi = p
        self.zi = z
        self.p = np.array(
            [
                100000.0,
                79432.82347243,
                63095.73444802,
                50118.72336273,
                39810.71705535,
                31622.77660168,
                25118.8643151,
                19952.62314969,
                15848.93192461,
                12589.25411794,
                10000.0,
                7943.28234724,
                6309.5734448,
                5011.87233627,
                3981.07170553,
                3162.27766017,
                2511.88643151,
                1995.26231497,
                1584.89319246,
                1258.92541179,
                1000.0,
            ]
        )
        self.z = np.array(
            [
                119.26605505,
                2118.57516973,
                4027.60345297,
                5862.87711004,
                7627.64498975,
                9297.02869728,
                10892.59889357,
                12434.63769365,
                13904.25695207,
                15290.83137241,
                16635.83815029,
                17963.99841403,
                19346.95698534,
                20750.73842791,
                22184.62422791,
                23675.44467966,
                25180.02964216,
                26780.47609985,
                28320.01292605,
                29820.7853969,
                31494.56521739,
            ]
        )

    def _interpolate(self, y):

        if not self.pi is None:
            xi = np.log10(self.pi)
            x = np.log10(self.p)
            yi = np.interp(xi, x, y)
            return yi

        if not self.zi is None:
            xi = self.zi
            x = self.z
            yi = np.interp(xi, x, y)
            return yi

        return y

    def get_pressure(self, *args, **kwargs):
        return self._interpolate(self.p)

    def get_temperature(self, *args, **kwargs):
        t = np.array(
            [
                298.98440367,
                287.22569932,
                276.81505687,
                264.51872336,
                252.79477857,
                241.6396106,
                230.84106763,
                220.73139122,
                210.94147842,
                201.7514298,
                195.60115607,
                198.65599366,
                204.08782794,
                209.70295371,
                215.04309815,
                218.4859783,
                221.8032664,
                225.38826646,
                228.7384274,
                231.92006504,
                235.52826087,
            ]
        )
        return self._interpolate(t)

    def get_altitude(self, *args, **kwargs):
        return self._interpolate(self.z)

    def get_CO2(self, *args, **kwargs):
        co2 = np.array(
            [
                0.0003685,
                0.0003685,
                0.0003685,
                0.0003685,
                0.0003685,
                0.0003685,
                0.0003685,
                0.0003685,
                0.00036846,
                0.00036807,
                0.00036701,
                0.00036548,
                0.00036395,
                0.000363,
                0.000363,
                0.000363,
                0.000363,
                0.000363,
                0.000363,
                0.000363,
                0.000363,
            ]
        )
        return self._interpolate(co2)

    def get_O2(self, *args, **kwargs):
        o2 = np.array(
            [
                0.20914768,
                0.20917247,
                0.20911265,
                0.20919441,
                0.20921843,
                0.20915963,
                0.20914215,
                0.20918673,
                0.20921144,
                0.20916823,
                0.20915074,
                0.20918311,
                0.20910791,
                0.20914558,
                0.20908499,
                0.20913025,
                0.20910134,
                0.20907593,
                0.20910521,
                0.2091797,
                0.20917443,
            ]
        )
        return self._interpolate(o2)

    def get_O3(self, *args, **kwargs):
        o3 = np.array(
            [
                2.90479819e-08,
                3.36389220e-08,
                3.56864241e-08,
                3.96190027e-08,
                4.38274934e-08,
                5.18099738e-08,
                6.50830255e-08,
                8.46137019e-08,
                1.03927382e-07,
                1.31159885e-07,
                2.11682715e-07,
                4.91257992e-07,
                1.10672315e-06,
                1.70139678e-06,
                2.58579126e-06,
                4.01127013e-06,
                5.57665027e-06,
                7.11467740e-06,
                8.29916685e-06,
                9.20149730e-06,
                9.63690492e-06,
            ]
        )
        return self._interpolate(o3)

    def get_H2O(self, *args, **kwargs):
        h2o = np.array(
            [
                2.51821658e-02,
                1.45514777e-02,
                4.41302228e-03,
                2.27390123e-03,
                9.60074728e-04,
                3.45144953e-04,
                8.58033141e-05,
                2.07439434e-05,
                6.57675492e-06,
                3.71219009e-06,
                2.93834588e-06,
                2.75786835e-06,
                2.60168486e-06,
                2.63916645e-06,
                2.81999017e-06,
                3.10463974e-06,
                3.27712882e-06,
                3.50106393e-06,
                3.73366696e-06,
                3.97511724e-06,
                4.18255652e-06,
            ]
        )
        return self._interpolate(h2o)

    def get_N2(self, *args, **kwargs):
        n2 = np.array(
            [
                0.78143085,
                0.78164177,
                0.78123672,
                0.78189026,
                0.78169185,
                0.78155673,
                0.78151964,
                0.78156695,
                0.78156223,
                0.78157601,
                0.78164542,
                0.78170006,
                0.78154427,
                0.78127151,
                0.78165497,
                0.78161547,
                0.7815775,
                0.78152597,
                0.78153412,
                0.78159145,
                0.78158467,
            ]
        )
        return self._interpolate(n2)

    def get_N2O(self, *args, **kwargs):
        n2o = np.array(
            [
                3.20250799e-07,
                3.20251239e-07,
                3.20164327e-07,
                3.20301300e-07,
                3.20289272e-07,
                3.19304895e-07,
                3.14651605e-07,
                3.07698606e-07,
                3.00687328e-07,
                2.92709822e-07,
                2.81926249e-07,
                2.67718964e-07,
                2.47226689e-07,
                2.23787977e-07,
                2.03619394e-07,
                1.90615464e-07,
                1.74473384e-07,
                1.63739574e-07,
                1.53277098e-07,
                1.42960728e-07,
                1.26706566e-07,
            ]
        )
        return self._interpolate(n2o)

    def get_surface_temperature(self, *args, **kwargs):
        return np.array([[290.0]])

    def get_surface_temperature(self, *args, **kwargs):
        return np.array([[290.0]])


class ArtsAtmosphere(DataProviderBase):
    def __init__(self, path, p):
        super().__init__()

        self.path = path
        self.pi = p

        files = Path(path).glob("*.xml")
        self.species = {}
        for filename in files:
            species_name = filename.name.split(".")[-2]
            self.species[species_name] = str(filename)

    def _interpolate(self, p, y):
        xi = np.log10(self.pi)
        x = np.log10(p)
        yi = np.interp(-xi, -x, y)
        return yi

    def get_pressure(self, *args, **kwargs):
        return self.pi

    def get_temperature(self, *args, **kwargs):
        filename = self.species["t"]
        data = load(filename)
        p = data.grids[0]
        return self._interpolate(p, data.data[:, 0, 0])

    def get_altitude(self, *args, **kwargs):
        filename = self.species["z"]
        data = load(filename)
        p = data.grids[0]
        return self._interpolate(p, data.data[:, 0, 0])

    def get_CO2(self, *args, **kwargs):
        filename = self.species["CO2"]
        data = load(filename)
        p = data.grids[0]
        return self._interpolate(p, data.data[:, 0, 0])

    def get_O2(self, *args, **kwargs):
        filename = self.species["O2"]
        data = load(filename)
        p = data.grids[0]
        return self._interpolate(p, data.data[:, 0, 0])

    def get_O3(self, *args, **kwargs):
        filename = self.species["O3"]
        data = load(filename)
        p = data.grids[0]
        return self._interpolate(p, data.data[:, 0, 0])

    def get_H2O(self, *args, **kwargs):
        filename = self.species["H2O"]
        data = load(filename)
        p = data.grids[0]
        return self._interpolate(p, data.data[:, 0, 0])

    def get_N2(self, *args, **kwargs):
        filename = self.species["N2"]
        data = load(filename)
        p = data.grids[0]
        return self._interpolate(p, data.data[:, 0, 0])

    def get_N2O(self, *args, **kwargs):
        filename = self.species["N2O"]
        data = load(filename)
        p = data.grids[0]
        return self._interpolate(p, data.data[:, 0, 0])

    def get_surface_temperature(self, *args, **kwargs):
        return np.array([[290.0]])
