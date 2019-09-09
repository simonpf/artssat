"""
This file tests the storing of forward simulation and retrieval
results to NetCDF files.
"""
import parts
import tempfile
import numpy as np
import shutil

import os
import sys
test_path = os.path.join(os.path.dirname(parts.__file__), "..", "tests")
sys.path.append(test_path)
from utils.setup import arts_simulation, arts_retrieval

def test_io_forward_simulation():
    """
    Test storing of forward simulation results to output file.
    """
    path = tempfile.mkdtemp()
    output_file = os.path.join(path, "output.nc")
    simulation = arts_simulation()
    simulation.setup(verbosity = 0)
    simulation.initialize_output_file(output_file, [("i", 1, 0), ("j", 2, 0)])
    simulation.run_ranges(range(1), range(2))

    fh      = simulation.output_file.file_handle
    v_names = [v for v in fh.variables]
    assert(v_names == ["y_" + s.name for s in simulation.sensors])
    for s in simulation.sensors:
        assert(np.all(np.isclose(fh.variables["y_" + s.name][-1, -1, :], s.y.ravel())))

    shutil.rmtree(path)

def test_io_retrieval():
    """
    Test storing of retrieval results to output file.
    """
    path = tempfile.mkdtemp()
    path = "."
    output_file = os.path.join(path, "output.nc")

    simulation = arts_simulation()
    simulation.setup(verbosity = 0)
    simulation.run(verbosity = 0)
    y = np.copy(simulation.workspace.y)

    retrieval = arts_retrieval()
    retrieval.setup(verbosity = 0)
    retrieval.retrieval.y = y

    retrieval.initialize_output_file(output_file, [("i", 1, 0), ("j", 2, 0)])
    retrieval.run_ranges(range(1), range(2))

    fh      = retrieval.output_file.file_handle
    g_names = [g for g in fh.groups]
    assert(g_names == ["Retrieval"])

    g = fh["Retrieval"]
    v_names = [v for v in g.variables]
    v_names_r = ["y_" + s.name for s in simulation.sensors] \
                + ["yf_" + s.name for s in simulation.sensors] \
                + ["diagnostics", "O2"]

    #shutil.rmtree(path)
