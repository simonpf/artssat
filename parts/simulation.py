import numpy as np
from typhon.arts.workspace     import Workspace
from parts.sensor.sensor       import ActiveSensor, PassiveSensor
from parts.scattering.solvers  import ScatteringSolver, RT4, Disort
from parts.jacobian            import JacobianCalculation
from parts.retrieval           import RetrievalCalculation
from parts.io                  import OutputFile

class ArtsSimulation:

    def __init__(self,
                 atmosphere = None,
                 data_provider = None,
                 sensors = [],
                 scattering_solver = RT4(nstreams = 12)):

        self._atmosphere        = atmosphere
        self._data_provider     = data_provider
        self._sensors           = sensors
        self._scattering_solver = scattering_solver
        self._data_provider     = data_provider
        self._setup             = False
        self._workspace         = None
        self.output_file        = None

        self.jacobian  = JacobianCalculation()
        self.retrieval = RetrievalCalculation()

        try:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            size = self.comm.Get_size()
            self.parallel = size > 1
        except:
            self.parallel = False


    #
    # Properties
    #

    @property
    def atmosphere(self):
        return self._atmosphere

    @property
    def sensors(self):
        return self._sensors

    @property
    def active_sensors(self):
        return [s for s in self._sensors if isinstance(s, ActiveSensor)]

    @property
    def passive_sensors(self):
        return [s for s in self._sensors if isinstance(s, PassiveSensor)]

    @property
    def data_provider(self):
        return self._data_provider

    @data_provider.setter
    def data_provider(self, dp):
        self._data_provider = dp

    @property
    def scattering_solver(self):
        return self._scattering_solver

    @scattering_solver.setter
    def scattering_solver(self, s):
        if isinstance(s, ScatteringSolver):
            self._scattering_solver = s
        else:
            raise ValueError("The scattering solver must be an instance of the "
                             "abstract ScatteringSolver class.")

    @property
    def workspace(self):
        return self._workspace

    #
    # Radiative transfer calculations
    #

    def setup(self, verbosity = 1):

        self.verbosity  = verbosity
        self._workspace = Workspace(verbosity = verbosity)
        ws = self._workspace
        ws.execute_controlfile("general/general.arts")
        ws.execute_controlfile("general/continua.arts")
        ws.execute_controlfile("general/agendas.arts")
        ws.execute_controlfile("general/planet_earth.arts")

        ws.Copy(ws.ppath_agenda, ws.ppath_agenda__FollowSensorLosPath)
        ws.Copy(ws.ppath_step_agenda, ws.ppath_step_agenda__GeometricPath)
        ws.Copy(ws.iy_space_agenda, ws.iy_space_agenda__CosmicBackground)
        ws.Copy(ws.iy_surface_agenda, ws.iy_surface_agenda__UseSurfaceRtprop)
        ws.Copy(ws.iy_main_agenda, ws.iy_main_agenda__Emission)

        self.atmosphere.setup(ws)

        for s in self.sensors:
            s.setup(ws, self.atmosphere.scattering)

        self._setup = True

    def check_dimensions(self, f, name):
        s = f.shape

        err = "Provided atmospheric " + name + " field"
        err += " is inconsistent with the dimensions of the atmosphere."

        if len(s) != len(self.dimensions):
            raise Exception(err)
        if not all([i == j or j == 0 for i,j \
                    in zip(s, self.dimensions)]):
            raise Exception(err)

    def _run_forward_simulation(self, sensors = []):

        ws = self.workspace

        # Jacobian setup
        self.jacobian.setup(ws, self.data_provider, *self.args, **self.kwargs)

        # Run atmospheric checks
        self.atmosphere.run_checks(ws)

        i_active = []
        i_passive = []

        if len(sensors) == 0:
            sensors = self.sensors

        for i,s in enumerate(sensors):
            if isinstance(s, ActiveSensor):
                i_active += [i]
            if isinstance(s, PassiveSensor):
                i_passive += [i]


        if len(i_active) > 1:
            raise Exception("Currently at most one active sensor is "
                            "supported in a multi sensor observation.")

        i = 0
        y_index = 0

        if self.atmosphere.scattering:
            ws.pnd_fieldCalcFromParticleBulkProps()

        # Simulate active sensor
        if len(i_active) > 0:

            # TODO: Get around the need to fix this.
            ws.stokes_dim = s.stokes_dimension

            s = sensors[i_active[0]]

            if self.atmosphere.scattering:
                f = s.make_y_calc_function(append = False)
                f(ws)
            else:
                ws.y.value = s.y_min * np.ones(s.y_vector_length)
                ws.y_f = s.f_grid[0] * np.ones(s.y_vector_length)
                ws.y_pol = [0] * s.y_vector_length
                ws.y_pos = 0.0 * np.ones((s.y_vector_length,
                                          len(self.atmosphere.dimensions)))
                ws.y_geo = 0.0 * np.ones((s.y_vector_length, 5))
                ws.y_los = 0.0 * np.ones((s.y_vector_length,
                                          min(len(self.atmosphere.dimensions), 2)))

            i += 1
            s.y = np.copy(ws.y.value[y_index:].reshape((-1, 1)))
            y_index += ws.y.value.size


        # Simulate passive sensors
        for s in [sensors[i] for i in i_passive]:

            # TODO: Get around the need to fix this.
            ws.stokes_dim = s.stokes_dimension

            # Run scattering solver
            if self.atmosphere.scattering:
                f = self.scattering_solver.make_solver_call(s)
                f(ws)

            f = s.make_y_calc_function(append = i > 0,
                                       scattering = self.atmosphere.scattering)
            f(ws)

            s.y = np.copy(ws.y.value[y_index:].reshape((-1, s.stokes_dimension)))
            y_index = ws.y.value.size

            i += 1


    def _run_retrieval(self, *args, **kwargs):

        ws = self.workspace
        scattering = len(self.atmosphere.scatterers) > 0

        self.retrieval.setup(self.workspace, self.sensors, self.scattering_solver,
                             scattering, self.data_provider, *args, **kwargs)


        ws.y = self.retrieval.y

        self.atmosphere.run_checks(ws)

        ws.covmat_seAddBlock(
            block = self.data_provider.get_observation_error_covariance(*args,
                                                                        **kwargs)
        )

        ws.OEM(**self.retrieval.settings)

    def run(self, *args, **kwargs):

        if not self._setup:
            raise Exception("setup() member function must be executed before"
                            " a simulation can be run.")

        self.args   = args
        self.kwargs = kwargs

        ws = self.workspace

        self.atmosphere.get_data(ws, self.data_provider, *args, **kwargs)
        for s in self.sensors:
            s.get_data(ws, self.data_provider, *args, **kwargs)

        # Run retrieval calculation
        if len(self.retrieval.retrieval_quantities) > 0:
            scattering = len(self.atmosphere.scatterers) > 0

            self.retrieval.run(self, *args, **kwargs)
        else:

            self._run_forward_simulation()

    def _run_ranges_mpi(self, ranges, *args, callback = None, **kwargs):
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        r = ranges[0]
        n  = (r.stop - r.start) // r.step
        dn  = n // size

        if n < size:
            raise Exception("Need at least as many steps in outermost range as "
                            " processors. Otherwise deadlocks occur.")

        n0  = rank * dn
        rem = n % size
        if rank < rem:
            dn = dn + 1
        n0 += min(rank, rem)

        for i in range(r.start + n0, r.start + n0 + dn):
            self._run_ranges(ranges[1:], *args, i, **kwargs, callback = callback)


    def _run_ranges(self, ranges, *args, callback = None, **kwargs):
        if len(ranges) == 0:
            self.run(*args, **kwargs)

            if not callback is None:
                callback(self)
            else:
                if not self.output_file is None:
                    self.store_results()
        else:
            r = ranges[0]
            for i in r:
                self._run_ranges(ranges[1:], *args, i, **kwargs)


    def run_ranges(self, *args, mpi = None, callback = None, **kwargs):

        if mpi is None:
            parallel = self.parallel
        else:
            parallel = mpi

        ranges = list(args)
        if parallel:
            self._run_ranges_mpi(ranges, **kwargs, callback = callback)
        else:
            self._run_ranges(ranges, **kwargs, callback = callback)

    def run_checks(self):
        self.atmosphere.run_checks(self.workspace)

    def initialize_output_file(self,
                               filename,
                               dimensions,
                               mode = "w",
                               full_retrieval_output = True):
        self.output_file = OutputFile(filename,
                                      dimensions = dimensions,
                                      mode = mode,
                                      full_retrieval_output = full_retrieval_output)

    def store_results(self):
        if not self.output_file is None:
            self.output_file.store_results(self)
        else:
            raise Exception("The output file must be initialized before results"
                            " can be written to it.")
