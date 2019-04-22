"""
parts.io
========

The `parts.io` module provides routines for the storing of simulations results.
"""
from netCDF4 import Dataset
import numpy as np

class OutputFile:
    """
    Class to store results from an ARTS simulation to a NetCDF file.

    """
    def __init__(self,
                 filename,
                 dimensions = None,
                 mode = "wb",
                 floating_point_format = "f4",
                 full_retrieval_output = True):
        """
        Create output file to store simulation output to.

        Arguments:

            filename(str): Path of the output file.

            dimensions(list): List of tuples :code:`(s, e, o)` containing
                for each dimensions over which simulations will be produced
                the start index :code:`s`, the end index :code:`e` and
                the offset :code:`o`.

            mode(str): String describing the mode to open the output file.
        """
        try:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            size = self.comm.Get_size()
            self.parallel = size > 1
        except:
            self.parallel = False

        self.filename   = filename
        self.mode       = mode
        self.dimensions = dimensions
        self.f_fp       = floating_point_format

        self.full_retrieval_output = full_retrieval_output
        self.initialized = False

    def _initialize_dimensions(self):
        """
        Initialize compute dimension in the output file.
        """
        # indices
        for n, s, _ in self.dimensions:
            self.file_handle.createDimension(n, s)

    def _initialize_forward_simulation_output(self, simulation):
        """
        Initializes output file for forward simulation results.
        Create dict attribute :code:`variables` containing variables
        corresponding to each sensor name.
        """
        self.variables = {}
        root = self.file_handle
        indices = [n for n, _, _ in self.dimensions]

        for s in simulation.sensors:
            dim = s.name + "_channels"
            root.createDimension(dim, s.y_vector_length)
            v = root.createVariable("y_" + s.name, self.f_fp,
                                    dimensions = tuple(indices  + [dim]))
            self.variables[s.name] = v

    def _initialize_retrieval_output(self, simulation):
        """
        Initialize output file for results from a retrieval
        calculation.

        Arguments:

            simulation: ArtsSimulation object from which to store the results.

        """

        retrieval = simulation.retrieval
        args      = simulation.args
        kwargs    = simulation.kwargs

        #
        # Global dimensions
        #

        # z grid
        p = simulation.workspace.p_grid.value
        self.file_handle.createDimension("z", p.size)

        # oem diagnostices
        self.file_handle.createDimension("oem_diagnostics", 5)

        #
        # Result groups
        #

        self.groups = []
        indices = [n for n, _, _ in self.dimensions]

        if not type(retrieval.results) == list:
            results = [retrieval.results]
        else:
            results = retrieval.results

        for r in results:
            group = self.file_handle.createGroup(r.name)
            self.groups += [group]

            # Retrieval quantities.
            for rq in retrieval.retrieval_quantities:
                v = group.createVariable(rq.name, self.f_fp, dimensions = tuple(indices + ["z"]))

            # OEM diagnostics.
            v = group.createVariable("diagnostics", self.f_fp,
                                     dimensions = tuple(indices + ["oem_diagnostics"]))

            # Observations and fit.
            for s in r.sensors:
                i, j = r.sensor_indices[s.name]
                m = j - i
                d1 = s.name + "_channels"
                group.createDimension(d1, m)
                v = group.createVariable("y_" + s.name, self.f_fp,
                                              dimensions = tuple(indices  + [d1]))
                v = group.createVariable("yf_" + s.name, self.f_fp,
                                              dimensions = tuple(indices  + [d1]))

            if self.full_retrieval_output:


                n = simulation.workspace.x.value.size
                m = simulation.workspace.y.value.size

                group.createDimension("m", m)
                group.createDimension("n", n)

                group.createVariable("G", self.f_fp, dimensions = tuple(indices + ["n", "m"]))
                group.createVariable("A", self.f_fp, dimensions = tuple(indices + ["n", "n"]))
                group.createVariable("covmat_so", self.f_fp, dimensions = tuple(indices + ["n", "n"]))
                group.createVariable("covmat_ss", self.f_fp, dimensions = tuple(indices + ["n", "n"]))
                group.createVariable("jacobian", self.f_fp, dimensions = tuple(indices + ["m", "n"]))



    def initialize(self, simulation):
        """
        Initialize output file.

        This creates all necessary dimensions and variables in the NetCDF4
        output file. This function is run automatically before the first
        entry is stored in the file.
        """
        self.file_handle = Dataset(self.filename,
                                   mode = self.mode,
                                   parallel = self.parallel)

        self._initialize_dimensions()

        if len(simulation.retrieval.retrieval_quantities) > 0:
            self._initialize_retrieval_output(simulation)
        else:
            self._initialize_forward_simulation_output(simulation)

        self.initialized = True

    def _store_forward_simulation_results(self, simulation):

        args   = [a - o for a, (_, _, o) in zip(simulation.args, self.dimensions)]
        kwargs = simulation.kwargs

        for s in simulation.sensors:
            var = self.variables[s.name]
            y   = np.copy(s.y.ravel())
            var.__setitem__(list(args) + [slice(0, None)], y)

        retrieval = simulation.retrieval

    def _store_retrieval_results(self, simulation):

        retrieval = simulation.retrieval
        args   = [a - o for a, (_, _, o) in zip(simulation.args, self.dimensions)]
        kwargs = simulation.kwargs

        if not type(retrieval.results) == list:
            results = [retrieval.results]
        else:
            results = retrieval.results

        for g, r in zip(self.groups, results):
            #
            # Retrieved quantities
            #

            for rq in retrieval.retrieval_quantities:
                x = r.get_result(rq, interpolate = True)
                if x is None:
                    x = r.get_xa(rq, interpolate = True)
                x = rq.transformation.invert(x)
                var = g.variables[rq.name]
                var.__setitem__(list(args) + [slice(0, None)], x)

            #
            # OEM diagnostics.
            #
            var = g.variables["diagnostics"]
            var.__setitem__(list(args) + [slice(0, None)], r.oem_diagnostics)

            #
            # Observation and fit.
            #

            for s in r.sensors:
                i, j = r.sensor_indices[s.name]
                y  = r.y[i : j]
                yf = r.yf[i : j]

                name = "y_" + s.name
                var = g[name]
                var.__setitem__(list(args) + [slice(0, None)], y)

                name = "yf_" + s.name
                var = g[name]
                var.__setitem__(list(args) + [slice(0, None)], yf)

            #
            # Remaining retrieval output
            #

            if self.full_retrieval_output and r.oem_diagnostics[0] <= 2.0:
                ws = simulation.workspace

                var = g["A"]
                var.__setitem__(list(args) + [slice(0, None)] * 2, ws.avk.value)

                var = g["G"]
                var.__setitem__(list(args) + [slice(0, None)] * 2, ws.dxdy.value)

                var = g["covmat_ss"]
                var.__setitem__(list(args) + [slice(0, None)] * 2, ws.covmat_ss.value)

                var = g["covmat_so"]
                var.__setitem__(list(args) + [slice(0, None)] * 2, ws.covmat_so.value)

                var = g["jacobian"]
                var.__setitem__(list(args) + [slice(0, None)] * 2, ws.jacobian.value)

                    

    def store_results(self, simulation):

        # Initialize file structure
        if not self.initialized:
            self.initialize(simulation)

        if len(simulation.retrieval.retrieval_quantities) > 0:
            self._store_retrieval_results(simulation)
        else:
            self._store_forward_simulation_results(simulation)
