from netCDF4 import Dataset

class OutputFile:
    def __init__(self, filename, dimension_names = None, mode = "w"):
        try:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            size = self.comm.Get_size()
            self.parallel = size > 1
        except:
            self.parallel = False

        self.file_handle = Dataset(filename, mode = mode,
                                   parallel = self.parallel)
        self.retrieval_output_initialized   = False
        self.initialized_simulations = False
        self.dimension_names = dimension_names

    def initialize_retrieval_output(self, retrieval):
        sim = retrieval.results[0].simulation
        args   = sim.args
        kwargs = sim.kwargs

        #if self.parallel:
        #    self.comm.Barrier()

        #
        # Global dimensions
        #

        # indices
        indices = []
        for i, a in enumerate(sim.args):
            if not self.dimension_names is None:
                name = self.dimension_names[i]
            else:
                name = "index_" + str(i + 1)
            self.file_handle.createDimension(name, None)
            indices += [name]

        # z grid
        p = sim.workspace.p_grid.value
        self.file_handle.createDimension("z", p.size)

        # oem diagnostices
        self.file_handle.createDimension("callbacks", len(retrieval.callbacks))

        # oem diagnostices
        self.file_handle.createDimension("oem_diagnostics", 5)

        dimensions = ["callbacks"] + indices

        for rq in retrieval.retrieval_quantities:
            v = self.file_handle.createVariable(rq.name, "f8", dimensions = tuple(dimensions + ["z"]))
            if self.parallel:
                v.set_collective(True)

        # OEM diagnostics.
        v = self.file_handle.createVariable("diagnostics", "f8",
                                            dimensions = tuple(dimensions + ["oem_diagnostics"]))
        if self.parallel:
            v.set_collective(True)

        # Observations and fit.
        for s in retrieval.results[0].simulation.sensors:
            m = s.y_vector_length
            d1 = s.name + "_channels"
            self.file_handle.createDimension(d1, m)
            v = self.file_handle.createVariable("y_" + s.name, "f8",
                                            dimensions = tuple(dimensions  + [d1]))
            if self.parallel:
                v.set_collective(True)

            v = self.file_handle.createVariable("yf_" + s.name, "f8",
                                            dimensions = tuple(dimensions  + [d1]))
            if self.parallel:
                v.set_collective(True)

        self.retrieval_output_initialized = True

    def store_retrieval_results(self, retrieval):

        # Initialize file structure
        if not self.retrieval_output_initialized:
            self.initialize_retrieval_output(retrieval)

        sim = retrieval.results[0].simulation
        args   = sim.args
        kwargs = sim.kwargs

        for i, r in enumerate(retrieval.results):
            #
            # Retrieved quantities
            #

            for rq in retrieval.retrieval_quantities:
                x = r.get_result(rq, interpolate = True)
                if x is None:
                    x = r.get_xa(rq, interpolate = True)
                x = rq.transformation.invert(x)
                var = self.file_handle.variables[rq.name]
                var.__setitem__([i] + list(args) + [slice(0, None)], x)

            #
            # OEM diagnostics.
            #

            for i, r in enumerate(retrieval.results):
                var = self.file_handle.variables["diagnostics"]
                var.__setitem__([i] + list(args) + [slice(0, None)], r.oem_diagnostics)

            #
            # Observation and fit.
            #

            for s in r.simulation.sensors:

                if s in r.sensors:
                    i, j = r.sensor_indices[s.name]
                    y  = r.y[i : j]
                    yf = r.yf[i : j]
                else:
                    y  = np.nan
                    yf = np.nan

                name = "y_" + s.name
                var = self.file_handle[name]
                var.__setitem__([i] + list(args) + [slice(0, None)], y)

                name = "yf_" + s.name
                var = self.file_handle[name]
                var.__setitem__([i] + list(args) + [slice(0, None)], yf)
