from netCDF4 import Dataset

class OutputFile:
    def __init__(self, filename, dimension_names = None, mode = "m"):
        try:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            size = self.comm.Get_size()
            print("MPI size: ", size)

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

        if self.parallel:
            self.comm.Barrier()

        indices = []
        for i, a in enumerate(sim.args):
            if not self.dimension_names is None:
                name = self.dimension_names[i]
            else:
                name = "index_" + str(i + 1)
            self.file_handle.createDimension(name, None)
            indices += [name]

        self.file_handle.createDimension("oem_diagnostics", 5)
        self.file_handle.createDimension("callbacks", len(retrieval.callbacks))
        indices = ["callbacks"] + indices

        p = sim.workspace.p_grid.value
        self.file_handle.createDimension("z", p.size)

        for rq in retrieval.retrieval_quantities:
            v = self.file_handle.createVariable(rq.name, "f8", dimensions = tuple(indices + ["z"]))
            if self.parallel:
                v.set_collective(True)
        self.retrieval_output_initialized = True

        v = self.file_handle.createVariable("diagnostics", "f8",
                                            dimensions = tuple(indices + ["oem_diagnostics"]))
        if self.parallel:
            v.set_collective(True)

        if self.parallel:
            self.comm.Barrier()

    def store_retrieval_results(self, retrieval):
        if not self.retrieval_output_initialized:
            self.initialize_retrieval_output(retrieval)

        sim = retrieval.results[0].simulation
        args   = sim.args
        kwargs = sim.kwargs

        for rq in retrieval.retrieval_quantities:
            for i, r in enumerate(retrieval.results):

                x = r.get_result(rq, interpolate = True)
                if x is None:
                    x = r.get_xa(rq, interpolate = True)
                x = rq.transformation.invert(x)
                var = self.file_handle.variables[rq.name]
                var.__setitem__([i] + list(args) + [slice(0, None)], x)

        for i, r in enumerate(retrieval.results):
            var = self.file_handle.variables["diagnostics"]
            var.__setitem__([i] + list(args) + [slice(0, None)], r.oem_diagnostics)

