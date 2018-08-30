import scipy as sp
import numpy as np

from abc import ABCMeta, abstractmethod

from typhon.arts.workspace import arts_agenda
from typhon.arts.workspace.agendas import Agenda
from parts.sensor import ActiveSensor, PassiveSensor
from typhon.arts.workspace.methods import workspace_methods

wsm = workspace_methods

class Transformation(metaclass = ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def setup(self, ws):
        pass

    @abstractmethod
    def __call__(self, x):
        pass

class Log10(Transformation):
    def __init__(self):
        pass

    def setup(self, ws):
        ws.jacobianSetFuncTransformation(transformation_func = "log10")

    def __call__(self, x):
        return np.log10(x)

class Log(Transformation):
    def __init__(self):
        pass

    def setup(self, ws):
        ws.jacobianSetFuncTransformation(transformation_func = "log")

    def __call__(self, x):
        return np.log10(x)

class Identity(Transformation):
    def __init__(self):
        pass

    def setup(self, ws):
        pass

    def __call__(self, x):
        return x

class Jacobian:
    """
    The :class:`Jacobian` class tracks quantities for which to compute
    the Jacbian within an ARTS sim

    """
    def __init__(self):
        self.jacobian_quantities = []

    def add(self, jq):
        self.jacobian_quantities += [jq.jacobian]

    def setup(self, ws):

        if not self.jacobian_quantities:
            ws.jacobianOff()
        else:
            ws.jacobianInit()
            for jq in self.jacobian_quantities:
                jq.setup_jacobian(ws)
            ws.jacobianClose()

class Retrieval:
    """
    The :class:`Retrieval` takes care of the book-keeping around retrieval
    quantities in an ARTS simulation.
    """
    def __init__(self):

        self.retrieval_quantities = []
        self.y = None

        self.settings = {"method" : "lm",
                         "max_start_cost" : np.inf,
                         "x_norm" : np.zeros(0),
                         "max_iter" : 10,
                         "stop_dx" : 0.1,
                         "lm_ga_settings" : np.array([100.0, 5.0, 2.0, 1e6, 1.0, 1.0]),
                         "clear_matrices" : 0,
                         "display_progress" : 1}

    def add(self, rq):
        self.retrieval_quantities += [rq.retrieval]

    def setup(self, ws, sensors, scattering_solver, scattering,
              retrieval_provider, *args, **kwargs):

        if not self.retrieval_quantities:
            return None

        xa = []
        x0 = []

        ws.retrievalDefInit()

        for rt in self.retrieval_quantities:
            rt.setup_retrieval(ws, retrieval_provider, *args, **kwargs)

            xa += [rt.xa]

            if rt.x0 is None:
                x0 += [rt.xa]
            else:
                x0 += [rt.x0]


        ws.retrievalDefClose()

        xa = np.concatenate(xa)
        x0 = np.concatenate(x0)

        ws.x = x0
        ws.xa = xa

        s = sensors[0]
        ws.Copy(ws.sensor_los,  s._wsvs["sensor_los"])
        ws.Copy(ws.sensor_pos,  s._wsvs["sensor_pos"])
        ws.Copy(ws.sensor_time, s._wsvs["sensor_time"])

        #
        # Setup inversion iterate agenda
        #

        agenda = Agenda.create("inversion_iterate_agenda")

        @arts_agenda
        def debug_print(ws):
            ws.Print(ws.x, 0)

        agenda.append(debug_print)

        for i, rq in enumerate(self.retrieval_quantities):
            preps = rq.get_iteration_preparations(i)
            if not preps is None:
                agenda.append(preps)

        arg_list = sensors[0].get_wsm_args(wsm["x2artsStandard"])
        agenda.add_method(ws, wsm["x2artsStandard"], *arg_list)

        if scattering:
            agenda.add_method(ws, wsm["pnd_fieldCalcFromParticleBulkProps"])
        #agenda = Agenda.create("inversion_iterate_agenda")

        i_active = []
        i_passive = []
        for i,s in enumerate(sensors):
            if isinstance(s, ActiveSensor):
                i_active += [i]
            if isinstance(s, PassiveSensor):
                i_passive += [i]

        i = 0
        y_index = 0

        # Active sensor
        if len(i_active) > 0:
            s = sensors[i_active[0]]

            agenda.append(arts_agenda(s.make_preparation_function()))
            agenda.append(arts_agenda(s.make_y_calc_function(append = False)))

            i += 1

        # Passive sensor
        for s in [sensors[i] for i in i_passive]:

            agenda.append(arts_agenda(s.make_preparation_function()))
            # Scattering solver call
            if scattering:
                m = scattering_solver.solver_call
                agenda.add_method(ws, m, *s.get_wsm_args(m),
                                  **scattering_solver.solver_kwargs)

            agenda.append(arts_agenda(
                s.make_y_calc_function(append = i > 0,
                                       scattering = scattering)
            ))
            i += 1


        def iteration_finalize(ws):
            ws.Ignore(ws.inversion_iteration_counter)

            ws.Copy(ws.yf, ws.y)
            ws.jacobianAdjustAndTransform()

        agenda.append(arts_agenda(iteration_finalize))

        ws.inversion_iterate_agenda = agenda


