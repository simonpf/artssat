from netCDF4 import Dataset

class OutputFile:
    def __init__(self, filename, dimension_names = None, mode = "m"):
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
        self.file_handle.createDimension("oem_diagnostics", 5)

        #
        # Result groups
        #

        self.groups = []

        for r in retrieval.results:
            group = self.file_handle.createGroup(r.name)
            self.groups += [group]

            # Retrieval quantities.
            for rq in retrieval.retrieval_quantities:
                v = group.createVariable(rq.name, "f8", dimensions = tuple(indices + ["z"]))
                if self.parallel:
                    v.set_collective(True)

            # OEM diagnostics.
            v = group.createVariable("diagnostics", "f8",
                                     dimensions = tuple(indices + ["oem_diagnostics"]))
            if self.parallel:
                v.set_collective(True)

            # Observations and fit.
            for s in r.sensors:
                i, j = r.sensor_indices[s.name]
                m = j - i
                d1 = s.name + "_channels"
                group.createDimension(d1, m)
                v = group.createVariable("y_" + s.name, "f8",
                                              dimensions = tuple(indices  + [d1]))
                if self.parallel:
                    v.set_collective(True)
                v = group.createVariable("yf_" + s.name, "f8",
                                              dimensions = tuple(indices  + [d1]))
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

        for g, r in zip(self.groups, retrieval.results):
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

            for i, r in enumerate(retrieval.results):
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
