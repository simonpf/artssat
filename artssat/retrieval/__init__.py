"""
artssat.retrieval
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

So that a quantity can be retrieved, it must inherit from the
:class:`RetrievalQuantity` base class. In this case, it can simply be added to
the :code:`retrieval` attribute of a given simulation:

::

    simulation.retrieval.add(q)

This will add the quantity :code:`q` to the quantities which will be retrieved.
Moreover it will instantiate the :class:`retrieval_class` class of the quantity
:code:`q` and set the :code:`q.retrieval` attribute of :code:`q`.

It is important to note that :code:`simulation.retrieval` and :code:`q.retrieval`
are of different types: :code:`simulation.retrieval` is of type
:code:`RetrievalCalculation` and coordinates the interaction with the ARTS workspace
required for the retrieval. The attribute :code:`q.retrieval` of the retrieval quantity
holds the settings and retrieval results specific to the quantity :code:`q`.

Reference
=========
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import weakref

from abc import ABCMeta, abstractmethod, abstractproperty

from pyarts.workspace         import arts_agenda
from pyarts.workspace.agendas import Agenda
from pyarts.workspace.methods import workspace_methods
wsm = workspace_methods

from artssat.jacobian    import JacobianBase, JacobianQuantity, Transformation
from artssat.arts_object import ArtsObject, arts_property
from artssat.arts_object import Dimension as dim
from artssat.sensor      import ActiveSensor, PassiveSensor


################################################################################
# RetrievalBase
################################################################################

class RetrievalBase(ArtsObject, metaclass = ABCMeta):
    """
    Base class for quantity-specific retrieval classes.

    The quantity specific retrieval classes are meant to hold settings as well
    as retrieval results after a calculation.

    In order to retrieve any quantity the following must be given:

        1. The a priori mean state :code:`xa` of the retrieval quantity
           of the Gaussian distribution representing the a priori assumptions.

        2. A covariance matrix :code:`covariance` or its inverse,
           a precision matrix :code:`precision`, of the Gaussian
           distribution representing the a priori assumption on the retrieval
           quantities. It is also possible to specify both, in which case
           the user has to make sure that they are consistent.

    In addition the following attributes can be given to customize the
    behaviour of the retrieval:

        3. An initial state :code:`x0` to start the retrieval iteration.

        4. A lower limit :code:`limit_low` which is used in each iteration
           as a lower cutoff for the next iteration step $x_i$.

        5. An upper limit :code:`limit_high` which is used in each iteration
           as a lower cutoff for the next iteration step $x_i$.

    .. note: All numeric values given for a retrieval quantity are assumed
        to be in the transformed space of the quantity. If a quantity has
        constant a priori mean :math:`x_a = 10^{-5}` and is
        :math:`log_{10}`-transformed, :code:`xa` should be given as :code:`-5`.

    """
    @arts_property(["Sparse, Matrix"], shape = (dim.Joker, dim.Joker), optional = True)
    def covariance_matrix(self):
        """
        The covariance matrix for the retrieval quantity.

        Each quantity must have at least one of the :code:`covariance_matrix`
        or the :code:`precision_matrix` attributes set.
        """
        return None

    @arts_property(["Sparse, Matrix"], shape = (dim.Joker, dim.Joker), optional = True)
    def precision_matrix(self):
        """
        The inverse of the covariance matrix.

        Each quantity must have at least one of the :code:`covariance_matrix`
        or the :code:`precision_matrix` attributes set.
        """
        return None

    @arts_property("Vector", shape = (dim.Joker,))
    def xa(self):
        """
        The mean of the Gaussian a priori distribution assumed for the
        quantity.
        """
        return None

    @arts_property("Vector", shape = (dim.Joker,), optional = True)
    def x0(self):
        """
        Optional start value for the retrieval iteration.
        """
        return None

    @arts_property("Numeric", optional = True)
    def limit_low(self):
        """
        Optional lower cutoff to apply to an iteration state :math:`x_i` before
        performing the  the forward simulation and also the backward
        transformation of the retrieval quantity.
        """
        return None

    @arts_property("Numeric", optional = True)
    def limit_high(self):
        """
        Optional upper cutoff to apply to an iteration state :math:`x_i` before
        performing the  the forward simulation and also the backward
        transformation of the retrieval quantity.
        """
        return None

    def get_xa(self, data_provider, *args, **kwargs):
        """
        Get a priori vector from data provider and set the :code:`xa`
        attribute this object.

        Arguments:

            data_provider: Data provide from which to query a priori
                state.

            *args, **kwargs: Arguments and keyword arguments to be passed
                on to the data provider.
        """
        try:
            fname = "get_" + self.quantity.name + "_xa"
            xa_fun = getattr(data_provider, fname)
            xa = xa_fun(*args, **kwargs)
            self.xa = xa_fun(*args, **kwargs)

        except AttributeError:
            raise Exception("The data provider must provide a get method for "
                           "the a priori state of retrieval quantity {0}."
                           .format(self.quantity.name))

    def setup(self, ws, data_provider, *args, **kwargs):
        """
        Generic setup method for retrieval quantities.

        This method tries to get the a priori vector :math:`x_a`, the start vector
        :math:`x_0`, the covariance and the precision matrix from the data
        provider and sets them in the given workspace.

        If no start vector :math:`x_0` is provided, the a priori state is used
        as initial state for the retrieval iteration.

        Of the covariance or precision matrices only one must be specified.

        Arguments:

            ws(:class:`pyarts.workspace.Workspace`): The ARTS workspace on
                which to setup the retrieval.

            data_provider: Data provider object to query a priori settings from.

            :code:`*args` and :code:`**kwargs` are forwarded to the data provider.
        """

        #
        # Get a priori and start values.
        #

        self.get_xa(data_provider, *args, **kwargs)

        fname = "get_" + self.quantity.name + "_x0"
        if hasattr(data_provider, fname):
            x0_fun = getattr(data_provider, fname)
            self.x0 = x0_fun(*args, **kwargs)
        else:
            self.x0 = np.copy(self.xa)

        self.get_data(ws, data_provider, *args, **kwargs)

        self.add(ws)

        #
        # Get covariance and precision matrix.
        #

        fname = "get_" + self.quantity.name + "_covariance"
        try:
            covmat_fun = getattr(data_provider, fname)
            covmat = covmat_fun(*args, **kwargs)
        except AttributeError as e:
            covmat = None
        except Exception as e:
            raise e

        fname = "get_" + self.quantity.name + "_precision"
        try:
            precmat_fun = getattr(data_provider, fname)
            precmat = precmat_fun(*args, **kwargs)
        except AttributeError as e:
            precmat = None
        except Exception as e:
            raise e

        if covmat is None and precmat is None:
            raise Exception("The data provider must provide a get method for "
                            "the covariance or the precision matrix of retrieval"
                            "quantity {0}." .format(self.quantity.name))

        self.quantity.transformation.setup(ws, data_provider, *args, **kwargs)

        if not covmat is None:
            ws.covmat_sxAddBlock(block = covmat)
        if not precmat is None:
            ws.covmat_sxAddInverseBlock(block = precmat)

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

################################################################################
# RetrievalQuantity
################################################################################

class RetrievalQuantity(JacobianQuantity):
    """
    Abstract interface for quantities for which can be retrieved using ARTS.

    Instances of the :code:`RetrievalQuantity` class can be added to the
    retrieval quantities of a simulation, which means that artssat will try
    to retrieve their value from a given observation.

    The interface for retrieval quantities is implemented by four properties
    that are shared by all quantities that can be retrieved with ARTS:

    1. :code:`transformation`: Retrieval quantities can be retrieved not only
        in the units used inside ARTS but also as transformed quantities, e.g.
        in :math:`log_{10}` space. The transformation property hols for each
        retrieval quantity a :class:`artssat.jacobian.Transformation` object
        representing the applied transformation.

    2. :code:`fixed`: Property indicating whether a retrieval quantity should
        be held fixed for a given retrieval run.

    3. :code:`retrieval_class`: A python class object holding the
        quantity-specific retrieval class that is instantiated and set as the
        quantities' :code:`retrieval` attribute when the quantity is added as
        retrieval quantity.

    4. :code:`retrieval`: Instance of the quantities :code:`retrieval_class`
        holding the retrieval settings and results specific to the quantity
        and a given retrieval run.
    """

    def __init__(self):
        self._fixed          = None
        self._retrieval      = None
        super().__init__()

    @property
    def fixed(self):
        """
        Whether or not the retrieval quantity should be held fixed for a given
        retrieval run.
        """
        return self._fixed

    @fixed.setter
    def fixed(self, t):
        self._fixed = t

    @abstractmethod
    def set_from_x(self, ws, xa):
        """
        Set method that sets the value of the retrieval quantity to a given
        a priori state.
        """
        pass

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

class RetrievalRun:
    """
    The :class:`RetrievalRun` represents a single call to the ARTS OEM
    workspace method. A single retrieval calculation can consist of
    several sequential retrieval runs performed with different settings.
    Each of these runs is represented by a single :code:`RetrievalRun`
    instance that holds the retrieval settings as well as the results
    of the single run.

    Attributes:

        name(:code:`str`): A name to identity the retrieval run.

        simulation(:code:`artssat.ArtsSimulation`): The simulation instance
            on which this retrieval run is executed.

        y(:code:`y`): The observation vector for the retrieval.

        settings(:code:`dict`): A dictionary holding the retrieval
            settings.
    """
    def __init__(self,
                 name,
                 simulation,
                 y,
                 settings,
                 sensor_indices,
                 retrieval_quantities,
                 previous_run = None):

        for rq in retrieval_quantities:
            rq.fixed = False

        self.name           = name
        self.y              = np.copy(y)
        self.settings       = settings
        self.sensors        = simulation.sensors.copy()
        self.sensor_indices = sensor_indices
        self.rq_indices     = {}
        self.retrieval_quantities = retrieval_quantities.copy()
        self.previous_run   = previous_run

        self._simulation = weakref.ref(simulation)
        self._data_provider = weakref.ref(simulation.data_provider)

        self.x = None

    @property
    def simulation(self):
        simulation = self._simulation()
        if simulation:
            return simulation
        else:
            raise ValueError("The corresponding simulation has "
                                    " been deleted.")

    @property
    def data_provider(self):
        data_provider = self._data_provider()
        if data_provider:
            return data_provider
        else:
            raise ValueError("The corresponding data provider has "
                                    " been destroyed.")

    def get_result(self, q, attribute = "x", interpolate = False, transform_back = False):

        if q in self.retrieval_quantities:
            i, j = self.rq_indices[q]
            x = getattr(self, attribute)
            if x is None:
                return x
            x = x[i : j]

            if transform_back:
                x = q.transformation.invert(x)

            if interpolate:
                ws = self.simulation.workspace
                grids = [ws.p_grid.value, ws.lat_grid.value, ws.lon_grid.value]
                grids = [g for g in grids if g.size > 0]
                x = q.retrieval.interpolate_to_grids(x, grids)

            return x

        if not self.previous_run is None:
            return self.previous_run.get_result(q, attribute = attribute)
        else:
            return None

    def get_xa(self, q, interpolate = True, transform_back = False):

        if transform_back:
            xa = q.transformation.invert(q.retrieval.xa)
        else:
            xa = q.retrieval.xa

        if interpolate:
            ws = self.simulation.workspace
            grids = [ws.p_grid.value, ws.lat_grid.value, ws.lon_grid.value]
            grids = [g for g in grids if g.size > 0]
            xa = q.retrieval.interpolate_to_grids(xa, grids)

        return xa

    def get_avk(self, q):
        if q in self.retrieval_quantities:
            i, j = self.rq_indices[q]
            x = getattr(self, "avk")
            return x[i : j, i : j]

        if not self.previous_run is None:
            return self.previous_run.get_result(q, attribute = "avk")
        else:
            return None

    #
    # Plotting functions
    #

    def plot_result(self,
                    q,
                    ax = None,
                    transform_back = True,
                    include_prior = True,
                    data_provider = None,
                    args = [],
                    kwargs = {}):
        """
        Plot retrieved results of given quantity.

        Works only in 1-dimensional atmospheres.

        Args:
            q: The retrieval quantity of which to plot the results
            ax: matplotilib Axes object in which to plot the results. If
                 None, a new axes object is constructed using suplots(1, 1).
            transform_back: Whether or not to plot results in
                transformed (:code:`False`) or original state space
                (True, default)
            include_prior: If :code:`True` also the a priori mean is plotted
                using a dashed line.
        """

        x = self.get_result(q,
                            interpolate = True,
                            transform_back = transform_back)
        if x is None:
            s = "No result for retrieval quantity {} available.".format(q.name)
            raise Exception(s)

        if ax is None:
            _, ax = plt.subplots(1, 1)

        z = self.simulation.workspace.z_field.value.ravel()

        ls = ax.plot(x, z, label = q.name)

        if include_prior:
            xa = self.get_xa(q,
                             interpolate = True,
                             transform_back = transform_back)
            ax.plot(xa, z, label = q.name, c = ls[0].get_color(), ls = "--")

        if data_provider:
            getter = getattr(data_provider, "get_" + q.name)
            x = getter(*args, **kwargs)
            ax.plot(x, z, label = "Reference", c = ls[0].get_color(), ls = "-.")

        return ax

    def plot_jacobian(self,
                      sensor,
                      q,
                      ax = None):
        """
        Plot the Jacobian of observations from a given sensor w.r.t.
        a given retrieval quantity. Note that the Jacobian is always
        displayed in transformed coordinates.

        Works only in 1-dimensional atmospheres.

        Args:
            sensor: The sensor providing the osbervations of which the Jacobian
                should be plotted.
            q: The retrieval quantity w.r.t. to which the Jacobian should be plotted.
            ax: matplotilib Axes object in which to plot the results. If
                 None, a new axes object is constructed using suplots(1, 1).
        """
        i1, j1 = self.sensor_indices[sensor.name]
        i2, j2 = self.rq_indices[q]
        dydx = self.jacobian[i1 : j1, i2: j2]
        print(dydx.shape)

        if dydx is None:
            s = "No result for retrieval quantity {} available.".format(q.name)
            raise Exception(s)

        if ax is None:
            _, ax = plt.subplots(1, 1)

        for i in range(dydx.shape[0]):
            ax.plot(dydx[i, :], label = "Channel {}".format(i))
        return ax

    def plot_a_priori_errors(self, *args, **kwargs):
        """
        Plots contributions of different components of the x-vector to the OEM
        cost.
        """
        i = 0
        for q in self.retrieval_quantities:
            _, ax = plt.subplots(1, 1)

            xa = q.retrieval.xa
            x = self.get_result(q,
                                interpolate = False,
                                transform_back = False)
            dx = x - xa

            data_provider = self.simulation.data_provider
            fname = "get_" + q.name + "_covariance"
            try:
                covmat_fun = getattr(data_provider, fname)
                covmat = covmat_fun(*args, **kwargs)
            except AttributeError as e:
                covmat = None
            except Exception as e:
                raise e

            fname = "get_" + q.name + "_precision"
            try:
                precmat_fun = getattr(data_provider, fname)
                precmat = precmat_fun(*args, **kwargs)
            except AttributeError as e:
                precmat = None
            except Exception as e:
                raise e

            if precmat:
                c = dx * np.dot(precmat, dx)
            else:
                c = dx * np.linalg.solve(covmat, dx)

            ax.plot(c)
            ax.set_title(q.name, loc = "left")

    def plot_observation_errors(self):
        """
        Plots contributions of different components of the x-vector to the OEM
        cost.
        """
        data_provider = self.simulation.data_provider
        get_name = "get_observation_error_covariance"
        f = getattr(data_provider, get_name)
        covmat_se = f(*args, **kwargs)

        for s in self.sensors:
            _, ax = plt.subplots(1, 1)
            i, j = self.sensor_indices[s.name]
            y = self.y[i : j]
            yf = self.yf[i : j]
            dy = yf - y
            s = covmat_se[i : j, i : j]
            c = dy * np.dot(s, dy)
            ax.plot(c)
            ax.set_title(s.name, loc = "left")

    def setup_a_priori(self, *args, **kwargs):
        """
        Setup the retrieval calculation.

        This methods performs the setup necessary to run the retrieval
        calculations on the given workspace. That means it does:

        1. Gather a priori mean states and covariance matrices for all
            registered retrieval quantities.

        2. Define the retrieval inside the given ARTS workspace.

        3. Construct the ARTS inversion iterate agenda.

        Arguments:

            ws(:code:`arts`): The :code:`pyarts.workspace.Workspace`
                object which to setup for the retrieval.

            sensors(list): The sensors to use for the retrieval calculation.

            scattering(:code:`bool`): Whether or not to the forward calculations
                involve scattering.

            retrieval_provider: The data provider providing the a priori data
                for the retrieval.

            *args, **kwargs: Arguments and keyword arguments that are passed on
                to the data provider.
        """

        ws            = self.simulation.workspace
        data_provider = self.data_provider

        xa = []
        x0 = []

        ws.retrievalDefInit()

        #
        # Need to store indices of retrieval quantities in x
        # for future reference.
        #

        rq_index = 0

        for rq in self.retrieval_quantities:

                rq.retrieval.setup(ws, data_provider, *args, **kwargs)

                xa += [rq.retrieval.xa]

                if not self.previous_run is None:
                    r = self.previous_run.get_result(rq)
                else:
                    r = None

                if not r is None:
                    x0 += [r]
                else:
                    if rq.retrieval.x0 is None:
                        x0 += [rq.retrieval.xa]
                    else:
                        x0 += [rq.retrieval.x0]

                self.rq_indices[rq] = (rq_index, rq_index + xa[-1].size)
                rq_index += xa[-1].size

        #
        # Set values of retrieval quantities excluded from retrieval.
        #

        for rq in self.simulation.retrieval.retrieval_quantities:

            # Need to setup transformation in case RQ has not yet been
            # retrieved.
            t = rq.transformation
            if hasattr(t, "initialize"):
                t.initialize(data_provider, *args, **kwargs)

            # Get data for retrieval quantity.
            rq.retrieval.get_data(ws, data_provider, *args, **kwargs)

            # Get result from previous run (x_p).
            # If not available set to a priori.
            x_p = self.get_result(rq)
            if x_p is None:
                rq.retrieval.get_xa(data_provider, *args, **kwargs)
                x_p = rq.retrieval.xa
            rq.set_from_x(ws, x_p)

        ws.retrievalDefClose()

        self.xa = np.concatenate(xa)
        self.x0 = np.concatenate(x0)

        ws.xa = self.xa
        ws.x  = self.x0

        #
        # We extract from the observation error covariance matrix
        # the artssat that are required for the used sensors.
        #

        get_name = "get_observation_error_covariance"
        f = getattr(data_provider, get_name)
        covmat_se = f(*args, **kwargs)

        if sp.sparse.issparse(covmat_se):
            covmat_se = covmat_se.todense()

        covmat_blocks = []

        for s in self.sensors:
            if isinstance(s, ActiveSensor):
                (i, j) = self.sensor_indices[s.name]
                covmat_blocks += [covmat_se[i:j, i:j]]

        for s in self.sensors:
            if not isinstance(s, ActiveSensor):
                (i, j) = self.sensor_indices[s.name]
                covmat_blocks += [covmat_se[i:j, i:j]]

        covmat_se = sp.sparse.block_diag(covmat_blocks, format = "coo")
        ws.covmat_seAddBlock(block = covmat_se)

    def setup_iteration_agenda(self):

        debug = self.simulation.retrieval.debug_mode

        if debug:
            self.debug = {"x"               : [],
                          "yf"              : [],
                          "jacobian"        : [],
                          "iteration_index" : []}

        ws            = self.simulation.workspace
        data_provider = self.data_provider

        s = self.sensors[0]
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
        #agenda.append(debug_print)

        for i, rq in enumerate(self.retrieval_quantities):
            preps = rq.retrieval.get_iteration_preparations(i)
            if not preps is None:
                agenda.append(preps)

        arg_list = self.sensors[0].get_wsm_kwargs(wsm["x2artsAtmAndSurf"])
        agenda.add_method(ws, wsm["x2artsAtmAndSurf"], **arg_list)

        scattering = len(self.simulation.atmosphere.scatterers) > 0
        if scattering:
            agenda.add_method(ws, wsm["pnd_fieldCalcFromParticleBulkProps"])

        i_active = []
        i_passive = []
        for i,s in enumerate(self.sensors):
            if isinstance(s, ActiveSensor):
                i_active += [i]
            if isinstance(s, PassiveSensor):
                i_passive += [i]

        i = 0
        y_index = 0

        # Active sensor
        if len(i_active) > 0:
            s = self.sensors[i_active[0]]

            agenda.append(arts_agenda(s.make_y_calc_function(append = False)))

            i += 1

        atmosphere = self.simulation.atmosphere
        # Passive sensor
        for s in [self.sensors[i] for i in i_passive]:

            # Scattering solver call
            if scattering:
                scattering_solver = self.simulation.scattering_solver
                agenda.append(arts_agenda(scattering_solver.make_solver_call(atmosphere, s)))

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

        @arts_agenda
        def get_debug(ws):
            self.debug["x"]               += [np.copy(ws.x.value)]
            self.debug["yf"]              += [np.copy(ws.yf.value)]
            self.debug["jacobian"]        += [np.copy(ws.jacobian.value)]
            self.debug["iteration_index"] += [ws.inversion_iteration_counter.value]

        if debug:
            agenda.append(get_debug)

        ws.inversion_iterate_agenda = agenda


    def run(self, *args, **kwargs):

        ws = self.simulation.workspace

        self.setup_iteration_agenda()
        self.setup_a_priori(*args, **kwargs)

        y_blocks = []
        for s in self.sensors:
            if isinstance(s, ActiveSensor):
                i, j = self.sensor_indices[s.name]
                y_blocks += [self.y[i : j]]
        for s in self.sensors:
            if not isinstance(s, ActiveSensor):
                i, j = self.sensor_indices[s.name]
                y_blocks += [self.y[i : j]]

        try:

            y = np.concatenate(y_blocks)
            self._y = y
            ws.y  = y
            ws.yf       = []
            ws.jacobian = []

            self.simulation.run_checks()
            ws.OEM(**self.settings)

        except Exception as e:
            ws.oem_diagnostics = 9 * np.ones(5)
            ws.yf       = None
            ws.jacobian = None
            ws.oem_errors = ["Error in OEM computation.", str(e)]

        self.x               = np.copy(ws.x.value)
        self.oem_diagnostics = np.copy(ws.oem_diagnostics)
        self.yf              = np.copy(ws.yf.value)
        self.jacobian        = np.copy(ws.jacobian.value)

        if self.oem_diagnostics[0] <= 2.0:
            ws.avkCalc()
            self.dxdy      = np.copy(ws.dxdy)
            self.jacobian  = np.copy(ws.jacobian)
            self.avk       = np.copy(ws.avk)
            ws.covmat_soCalc()
            self.covmat_so = np.copy(ws.covmat_so.value)
            ws.covmat_ssCalc()
            self.covmat_ss = np.copy(ws.covmat_ss.value)
        else:
            self.dxdy      = None
            self.jacobian  = None
            self.avk       = None
            self.covmat_so = None
            self.covmat_ss = None

        if self.oem_diagnostics[0] == 9.0:
            try:
                self.oem_errors = ws.oem_errors.value
                print("Error in OEM calculation:")
                print(self.oem_errors)
            except Exception as e:
                self.oem_errors = ["Error in OEM computation.", str(e)]

        else:
            self.oem_errors = None


class RetrievalCalculation:
    """
    The :class:`Retrieval` takes care of the book-keeping around retrieval
    quantities in an ARTS simulation as well as the execution of the retrieval
    calculation.

    Arguments:

        debug_mode(:code:`bool`): If set to true, debug information will be
            collected while the retrieval is run.
    """
    def __init__(self,
                 debug_mode = False):

        self.retrieval_quantities = []
        self.y = None

        self.settings = {"method" : "lm",
                         "max_start_cost" : np.inf,
                         "x_norm" : np.zeros(0),
                         "max_iter" : 20,
                         "stop_dx" : 0.1,
                         "lm_ga_settings" : np.array([1000.0, 5.0, 2.0, 1e6, 1.0, 1.0]),
                         "clear_matrices" : 0,
                         "display_progress" : 1}

        self.callbacks = []
        self.debug_mode = debug_mode

    def add(self, rq):
        """
        Add a retrieval quantity to the retrieval calculation.

        This registers an atmospheric quantity as a retrieval quantity,
        which means that artssat will try to retrieve the quantity using ARTS's
        OEM method instead of querying its value from the data provider. While
        the data provider is not required to provide get methods for
        the retrieval quantity itself, it must provide values for its a priori
        mean and covariance matrix.

        Arguments:

            rq(:code:`RetrievalQuantity`): The retrieval quantity for which to
                compute the retrieval.

        """
        rq.retrieval = rq.retrieval_class(rq, len(self.retrieval_quantities))
        self.retrieval_quantities += [rq]


    def _get_y_vector(self, simulation, *args, **kwargs):

        if len(simulation.sensors) == 0:
            raise Exception("Can't perform retrieval without sensors.")

        y = getattr(self, "y", None)
        if y is None:
            f = getattr(simulation.data_provider, "get_y", None)
            if not f is None:
                y = f(*args, **kwargs)
            else:
                ys = []

                for ss in [simulation.active_sensors, simulation.passive_sensors]:
                    for s in ss:
                        fname = "get_y_" + s.name
                        f = getattr(simulation.data_provider, fname, None)
                        if f is None:
                            raise Exception("No measurement vector provided for "
                                            " sensor {0}.".format(s.name))
                        ys += [f(*args, **kwargs).ravel()]
                y = np.concatenate(ys)
        return y

    def run(self, simulation, *args, **kwargs):
        """
        Setup the retrieval calculation.

        This methods performs the setup necessary to run the retrieval
        calculations on the given workspace. That means it does:

        1. Gather a priori mean states and covariance matrices for all
            registered retrieval quantities.

        2. Define the retrieval inside the given ARTS workspace.

        3. Construct the ARTS inversion iterate agenda.

        Arguments:

            ws(:code:`arts`): The :code:`pyarts.workspace.Workspace`
                object which to setup for the retrieval.

            sensors(list): The sensors to use for the retrieval calculation.

            scattering(:code:`bool`): Whether or not to the forward calculations
                involve scattering.

            retrieval_provider: The data provider providing the a priori data
                for the retrieval.

            *args, **kwargs: Arguments and keyword arguments that are passed on
                to the data provider.
        """

        # Determine sensor indices of y vector
        i_start = 0
        self.sensor_indices = {}
        for s in simulation.active_sensors:
            s.get_data(simulation.workspace, simulation.data_provider, *args, **kwargs)
            self.sensor_indices[s.name] = (i_start, i_start + s.y_vector_length)
            i_start += s.y_vector_length
        for s in simulation.passive_sensors:
            self.sensor_indices[s.name] = (i_start, i_start + s.y_vector_length)
            i_start += s.y_vector_length

        self._y = self._get_y_vector(simulation, *args, **kwargs)

        ws   = simulation.workspace

        previous_run = None
        if self.callbacks == []:
            #
            # No retrieval callback provided. The retrieval consists only
            # of a single run.
            #
            retrieval = RetrievalRun("Retrieval",
                                     simulation,
                                     self._y,
                                     self.settings,
                                     self.sensor_indices,
                                     self.retrieval_quantities)
            retrieval.run(*args, **kwargs)
            self.results = retrieval
        else:
            #
            # If a list of retrieval callbacks is provided, an OEM run
            # is run for each callback. Forwarding retrieval results to
            # the next call.
            #
            self.results = []
            for cb in self.callbacks:

                if type(cb) is tuple:
                    name, cb = cb
                else:
                    name = str(len(self.results))

                retrieval = RetrievalRun(name,
                                         simulation,
                                         self._y,
                                         self.settings,
                                         self.sensor_indices,
                                         self.retrieval_quantities,
                                         previous_run = previous_run)

                if not cb is None:
                    cb(retrieval)

                retrieval.run(*args, **kwargs)
                self.results += [retrieval]
                previous_run = retrieval
