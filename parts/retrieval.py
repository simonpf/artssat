"""
parts.retrieval
-----------------

The :code:`retrieval` module handles retrieval calculations with ARTS.
The retrieval functionality is provided by a class structure that is
similar to the one used for Jacobians:

1. :class:`RetrievalCalculation` handles the actual calculation of
   retrievals through the ARTS workspace

2. :class:`RetrievalQuantity` defines the general interface for quantities
   which can be retrieved and how to add them to the list of quantities
   to retrieve.

3. :class:`RetrievalBase` defines an abstract base class for the
   retrieval classes that encapsulate the settings and code that
   is specific to the given quantity.

Retrieving quantities
=====================

To retrieve a quantity is must derive from :class:`RetrievalQuantity`. If this
is the case it can simply be added to the :code:`retrieval` attribute of a
given simulation:

    simulation.retrieval.add(q)

This will add the quantity :code:`q` to the quantities which will be retrieved.
Moreover it will instantiate the :class:`retrieval_class` class of the quantity
:code:`q` and set the :code:`q.retrieval` attribute of :code:`q`.

It is important to note that :code:`simulation.retrieval` and :code:`q.retrieval`
are of different types: :code:`simulation.retrieval` is of type
:code:`RetrievalCalculation` and handles the calculation of the Jacobian for
multiple quantities whereas :code:`q.retrieval` holds the settings and results
of the retrieval calculation for :code:`q`.

Reference
=========

"""

import numpy as np

from abc import ABCMeta, abstractmethod, abstractproperty

from typhon.arts.workspace         import arts_agenda
from typhon.arts.workspace.agendas import Agenda
from typhon.arts.workspace.methods import workspace_methods
wsm = workspace_methods

from parts.jacobian    import JacobianBase, JacobianQuantity, Transformation
from parts.arts_object import ArtsObject, arts_property
from parts.arts_object import Dimension as dim
from parts.sensor      import ActiveSensor, PassiveSensor


################################################################################
# RetrievalBase class
################################################################################

class RetrievalBase(ArtsObject, metaclass = ABCMeta):

    @arts_property(["Sparse, Matrix"], shape = (dim.Joker, dim.Joker))
    def covariance_matrix(self):
        return None

    @arts_property("Numeric", shape = (dim.Joker,))
    def xa(self):
        return None

    @arts_property("Numeric", shape = (dim.Joker,))
    def x0(self):
        return None

    @arts_property("Numeric")
    def limit_low(self):
        return None

    @arts_property("Numeric")
    def limit_high(self):
        return None

    def setup(self, ws, data_provider, *args, **kwargs):

        #
        # Get a priori and start values.
        #

        fname = "get_" + self.quantity.name + "_xa"
        xa_fun = getattr(data_provider, fname)
        self.xa = xa_fun(*args, **kwargs)

        fname = "get_" + self.quantity.name + "_x0"
        if hasattr(data_provider, fname):
            x0_fun = getattr(data_provider, fname)
            self.x0 = x0_fun(*args, **kwargs)
        else:
            self.x0 = np.copy(self.xa)

        fname = "get_" + self.quantity.name + "_covariance"
        covmat_fun = getattr(data_provider, fname)
        covmat = covmat_fun(*args, **kwargs)

        ws.covmat_block = covmat

        self.add(ws)
        self.quantity.transformation.setup(ws)

    def get_iteration_preparations(self, index):

        if self.limit_low is None and self.limit_high is None:
            return None

        limit_low = -np.inf
        if self._limit_low.fixed:
            limit_low = self._limit_low.value

        limit_high = np.inf
        if self._limit_high.fixed:
            limit_high = self._limit_high.value

        def agenda(ws):
            ws.xClip(ijq = index, limit_low = limit_low, limit_high = limit_high)

        return arts_agenda(agenda)


class RetrievalQuantity(JacobianQuantity):
    """
    Abstract interface for quantities for which a Jacobian can be computed.

    Quantities for which a Jacobian can be computed must expose a
    :code:`jacobian_class` which holds all quantity-specific WSM calls and
    settings required to compute its Jacobian.

    After a quantity has been added to the Jacobian quantities of a simulation,
    the :code:`jacobian_class` object representing the settings and results of
    the Jacobian calculation for this specific object can be accessed through
    its :code:`jacobian` property.
    """

    def __init__(self):
        self._transformation = None
        super().__init__()

    @property
    def transformation(self):
        return self._transformation

    @transformation.setter
    def transformation(self, t):
        if not isinstance(t, Transformation):
            raise TypeError("The transformation of a retrieval quantity must"\
                            "of type Transformation.")
        else:
            self._transformation = t

    @abstractproperty
    def retrieval_class(self):
        """
        Return the class object that holds the actual implementation of the
        Jacobian calculation.
        """
        pass

    @property
    def retrieval(self):
        """
        The :code:`jacobian_class` object holding the quantity-specific settings
        and actual results of the Jacobian calculations for this quantity.
        """
        return self._retrieval

    @retrieval.setter
    def retrieval(self, r):
        if not isinstance(r, self.retrieval_class):# or \
           #not isinstance(r, RetrievalBase):
            raise ValueError("The retrieval property of a RetrievalQuantity"\
                             " can only be set to an instance of the objects"\
                             "own retrieval_class.")
        else:
            self._retrieval = r


################################################################################
# RetrievalCalculation
################################################################################

class RetrievalCalculation:
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
        rq.retrieval = rq.retrieval_class(rq, len(self.retrieval_quantities))
        self.retrieval_quantities += [rq]

    def setup(self, ws, sensors, scattering_solver, scattering,
              retrieval_provider, *args, **kwargs):

        if not self.retrieval_quantities:
            return None

        xa = []
        x0 = []

        ws.retrievalDefInit()

        for rt in self.retrieval_quantities:
            rt.retrieval.setup(ws, retrieval_provider, *args, **kwargs)

            xa += [rt.retrieval.xa]

            if rt.retrieval.x0 is None:
                x0 += [rt.retrieval.xa]
            else:
                x0 += [rt.retrieval.x0]


        ws.retrievalDefClose()

        print(xa)
        print(x0)
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
            preps = rq.retrieval.get_iteration_preparations(i)
            if not preps is None:
                agenda.append(preps)

        arg_list = sensors[0].get_wsm_args(wsm["x2artsAtmAndSurf"])
        agenda.add_method(ws, wsm["x2artsAtmAndSurf"], *arg_list)

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

            agenda.append(arts_agenda(s.make_y_calc_function(append = False)))

            i += 1

        # Passive sensor
        for s in [sensors[i] for i in i_passive]:

            # Scattering solver call
            if scattering:
                agenda.append(arts_agenda(scattering_solver.make_solver_call(s)))

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

