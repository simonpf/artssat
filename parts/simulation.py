import numpy as np
from typhon.arts.workspace     import Workspace
from parts.sensor.sensor       import ActiveSensor, PassiveSensor
from parts.scattering.solvers  import ScatteringSolver, RT4
from parts.jacobian            import JacobianCalculation
from parts.retrieval           import RetrievalCalculation

class ArtsSimulation:
    def __init__(self,
                 atmosphere = None,
                 data_provider = None,
                 sensors = [],
                 scattering_solver = RT4()):

        self._atmosphere        = atmosphere
        self._data_provider     = data_provider
        self._sensors           = sensors
        self._scattering_solver = scattering_solver
        self._data_provider     = data_provider
        self._workspace         = None

        self.jacobian  = JacobianCalculation()
        self.retrieval = RetrievalCalculation()


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

    def setup(self):

        self._workspace = Workspace()
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

    def check_dimensions(self, f, name):
        s = f.shape

        err = "Provided atmospheric " + name + " field"
        err += " is inconsistent with the dimensions of the atmosphere."

        if len(s) != len(self.dimensions):
            raise Exception(err)
        if not all([i == j or j == 0 for i,j \
                    in zip(s, self.dimensions)]):
            raise Exception(err)

    def _run_forward_simulation(self):

        ws = self.workspace

        # Jacobian setup
        self.jacobian.setup(ws)

        # Run atmospheric checks
        self.atmosphere.run_checks(ws)

        i_active = []
        i_passive = []
        for i,s in enumerate(self.sensors):
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

            s = self.sensors[i_active[0]]

            f = s.make_y_calc_function(append = False)
            f(ws)

            i += 1
            s.y = np.copy(ws.y.value[y_index:].reshape((-1, 1)))
            y_index += ws.y.value.size


        # Simulate passive sensors
        for s in [self.sensors[i] for i in i_passive]:

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

    def run_checks(self):
        self.atmosphere.run_checks(self.workspace)
