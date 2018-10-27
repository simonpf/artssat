"""
This module contains test data required to run the parts tests.

The module looks for test data in the location specified by the
:code:`PARTS_TEST_DATA` environment variable.
"""
import os
import parts

if "PARTS_TEST_DATA" in os.environ:
    test_data_path  = os.environ["PARTS_TEST_DATA"]
else:
    parts_path = os.path.dirname(parts.__file__)
    test_data_path  = os.path.join(parts_path, "..", "tests", "data")

scattering_data = os.path.join(test_data_path,
                               "SectorSnowflake.xml")
scattering_meta = os.path.join(test_data_path,
                               "SectorSnowflake.meta.xml")
