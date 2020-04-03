"""
This module contains test data required to run the artssat tests.

The module looks for test data in the location specified by the
:code:`ARTSSAT_TEST_DATA` environment variable.
"""
import os
import artssat

if "ARTSSAT_TEST_DATA" in os.environ:
    test_data_path  = os.environ["ARTSSAT_TEST_DATA"]
else:
    artssat_path = os.path.dirname(artssat.__file__)
    test_data_path  = os.path.join(artssat_path, "..", "tests", "data")

scattering_data = os.path.join(test_data_path, "ice.xml")
scattering_meta = os.path.join(test_data_path, "ice.meta.xml")
