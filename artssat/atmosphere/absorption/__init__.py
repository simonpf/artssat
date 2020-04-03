"""
Absorption
==========

"""
from abc import ABCMeta, abstractmethod, abstractproperty
from artssat.atmosphere.atmospheric_quantity \
    import AtmosphericQuantity, extend_dimensions

from artssat.arts_object import ArtsObject, arts_property
from artssat.jacobian    import JacobianBase
from artssat.retrieval   import RetrievalBase, RetrievalQuantity

import numpy as np
from pyarts.workspace import arts_agenda
from typhon.physics.atmosphere import vmr2relative_humidity, \
    relative_humidity2vmr

################################################################################
# Retrieval units
################################################################################

class Unit(metaclass = ABCMeta):
    """
    Abstract base class for classes representing units used for the calculation
    of Jacobians and retrievals of absorption species.
    """
    def __init__():
        pass

    @abstractmethod
    def to_arts(self, ws, x):
        pass

    @abstractmethod
    def from_arts(self, ws, x):
        pass

    @abstractproperty
    def arts_name(self):
        pass


class VMR(Unit):
    """
    VMR is the default unit used for absorption species in ARTS. If this unit
    is used value from the state vector are plugged in as they are into
    the ARTS vmr field.
    """

    def __init__(self):
        pass

    def to_arts(self, ws, x):
        """
        Does nothing.
        """
        return x

    def from_arts(self, ws, x):
        """
        Does nothing.
        """
        return x

    @property
    def arts_name(self):
        return "vmr"

class Relative(Unit):
    """
    In relative units, the amount of a quantity is specified relative to a
    reference profile or field. 

    If this unit is used in a Jacobian calculation, then the Jacobian is
    calculated w.r.t. a relative perturbation.

    If this unit is used in the retrieval, values in the state vector are
    interpreted as multiplicative perturbations of the reference profile
    or field.
    """
    def __init__(self, x_ref):
        self.x_ref = x_ref

    def to_arts(self, ws, x):
        return self.x_ref * x

    def from_arts(self, ws, y):
        return y / self.x_ref

    @property
    def arts_name(self):
        return "rel"

class RelativeHumidity(Unit):
    """
    Relative humidity is available only for the retrieval of H2O.
    """
    def __init__(self):
        pass

    def to_arts(self, ws, rh):
        """
        Converts value given in relative humidity units to ARTS vmr units.

        Arguments:

            ws(:code:`pyarts.workspace.Workspace`): Workspace object
                which contains pressure grid and temperature field required
                for the converstion.

            rh(:code:`numpy.ndarray`): Relative humidity values to convert.

        Returns:

            :code:`numpy.ndarray` containing the converted RH values.

        """
        p   = ws.p_grid.value.reshape(-1, 1, 1)
        t   = ws.t_field.value
        vmr = relative_humidity2vmr(rh, p, t)
        return vmr

    def from_arts(self, ws, vmr):
        """
        Converts a value given in ARTS vmr units back to relative humidity.

        Arguments:

            ws(:code:`pyarts.workspace.Workspace`): Workspace object
                which contains pressure grid and temperature field required
                for the conversion.

            vmr(:code:`numpy.ndarray`): Values in ARTS vmr units to convert
                to relative humidity.

        Returns:

            :code:`numpy.ndarray` containing the converted RH values.

        """
        p  = ws.p_grid.value.reshape(-1, 1, 1)
        t  = ws.t_field.value
        rh = vmr2relative_humidity(vmr, p, t)
        return rh

    @property
    def arts_name(self):
        return "rh"

################################################################################
# The Jacobian class
################################################################################

class Jacobian(JacobianBase, ArtsObject):

    @arts_property("Numeric")
    def perturbation(self):
        return 0.01

    def __init__(self, quantity, index):

        JacobianBase.__init__(self, quantity, index)
        ArtsObject.__init__(self)

        self.unit = VMR()
        self.for_species_tag = 1

    def _make_setup_kwargs(self, ws):
        kwargs = self.get_grids(ws)
        kwargs.update({"species" : self.quantity.get_tag_string(),
                       "unit" : self.unit.arts_name,
                       "for_species_tag" : self.for_species_tag})
        return kwargs

    def setup(self, ws):
        kwargs = self._make_setup_kwargs(ws)
        ws.jacobianAddAbsSpecies(**kwargs)

class Retrieval(RetrievalBase, Jacobian):

    def __init__(self, quantity, index):
        super().__init__(quantity, index)

    def add(self, ws):
        ws.retrievalAddAbsSpecies(**self._make_setup_kwargs(ws))


class AbsorptionSpecies(AtmosphericQuantity, RetrievalQuantity):

    def __init__(self,
                 name,
                 from_catalog = False,
                 cia = None,
                 frequency_range = None,
                 isotopologues = None,
                 model = None,
                 on_the_fly = True,
                 zeeman = False,
                 lineshape = "no_shape",
                 normalization = "no_norm",
                 cutoff = -1,
                 cutoff_type = "ByBand"):


        AtmosphericQuantity.__init__(self,
                                     name,
                                     (0, 0, 0))
        RetrievalQuantity.__init__(self)

        self._dimensions = (0, 0, 0)

        self._from_catalog    = from_catalog
        self._cia             = cia
        self._frequency_range = frequency_range
        self._isotopologues   = isotopologues
        self._jacobian        = None
        self._model           = model
        self._on_the_fly      = on_the_fly
        self._retrieval       = None
        self._zeeman          = zeeman

        self._lineshape = lineshape
        self._normalization = normalization
        self._cutoff = cutoff
        self._cutoff_type = cutoff_type


    #
    # Abstract properties
    #

    def dimensions(self):
        return self._dimensions

    #
    # Properties
    #

    @property
    def from_catalog(self):
        return self._from_catalog

    @property
    def cia(self):
        return self._cia

    @property
    def isotopologues(self):
        return self._isotopologues

    @property
    def model(self):
        return self._model

    @property
    def frequency_range(self):
        return self._frequency_range

    @property
    def on_the_fly(self):
        return self._on_the_fly

    @property
    def zeeman(self):
        return self._zeeman

    @property
    def lineshape(self):
        return self._lineshape

    @property
    def cutoff(self):
        return self._cutoff

    @property
    def cutoff_type(self):
        return self._cutoff_type

    @property
    def normalization(self):
        return self._normalization

    def _get_tag_string(self,
                        zeeman = False,
                        isotopologue = None,
                        model = None,
                        frequency_range = None):

        ts = self._name
        ts += "-"

        if zeeman:
            ts += "Z"
            ts += "-"

        if not isotopologue is None:
            ts += isotopologues
            ts += "-"

        if not model is None:
            ts += model
            ts += "-"

        if not frequency_range is None:
           ts += frequency_range[0]
           ts += "-"
           ts += frequency_range[1]
        return ts

    def get_tag_string(self):

        tss = []

        if self.from_catalog:
            tss += [self._name]

        z = self._zeeman

        if type(self._isotopologues) is list:
            isotopologues = self._isotopologues
        else:
            isotopologues = [self._isotopologues]

        if type(self._model) is list:
            models = self._model
        else:
            models = [self._model]

        if type(self._frequency_range) is list:
            frequency_ranges = self._frequency_range
        else:
            frequency_ranges = [self._frequency_range]


        for i in isotopologues:
            for m in models:
                for fr in frequency_ranges:
                    tss += [self._get_tag_string(z, i, m, fr)]

        return ",".join(tss)

    #
    # Jacobian & retrieval
    #

    @property
    def jacobian_class(self):
        return Jacobian

    @property
    def retrieval_class(self):
        return Retrieval

    def set_from_x(self, ws, x):

        grids = [ws.p_grid.value, ws.lat_grid.value, ws.lon_grid.value]
        grids = [g for g in grids if g.size > 0]
        x = self.transformation.invert(x)
        x = self.retrieval.interpolate_to_grids(x, grids)
        x = x.reshape(ws.vmr_field.value.shape[1:])

        unit = self.retrieval.unit
        x = unit.to_arts(ws, x)

        if self._wsv_index is None:
            raise Exception("Absorber's wsv_index is unknown. This is likely "
                            "its setup(...) routine has not been called.")

        ws.vmr_field.value[self._wsv_index, :, :, :] = x

    #
    # Retrieval
    #

    @property
    def retrieved(self):
        return not self._retrieval is None

    #
    # Abstract methods
    #

    def setup(self, ws, i):
        self._wsv_index = i

    def get_data(self, ws, provider, *args, **kwargs):

        if not self.retrieved:

            dimensions = ws.t_field.shape
            f = provider.__getattribute__("get_" + self.name)
            x = f(*args, **kwargs)
            x = extend_dimensions(x)

            if not x.shape == dimensions:
                raise Exception("Shape of {0} field is inconsistent with "
                                "the dimensions of the atmosphere."
                                .format(self.name))

            ws.vmr_field.value[self._wsv_index, :, :, :] = x

class H2O(AbsorptionSpecies):
    def __init__(self,
                 from_catalog = False,
                 cia = None,
                 frequency_range = None,
                 isotopologues = None,
                 model = "PWR98",
                 on_the_fly = True,
                 zeeman = False,
                 lineshape = "VP",
                 normalization = "VVW",
                 cutoff = -1,
                 cutoff_type = "ByBand"):
        super().__init__("H2O",
                         from_catalog = from_catalog,
                         cia = cia,
                         frequency_range = frequency_range,
                         isotopologues = isotopologues,
                         model = model,
                         on_the_fly = on_the_fly,
                         zeeman = zeeman,
                         lineshape = lineshape,
                         normalization = normalization,
                         cutoff = cutoff,
                         cutoff_type = cutoff_type)

class N2(AbsorptionSpecies):
    def __init__(self,
                 from_catalog = False,
                 cia = None,
                 frequency_range = None,
                 isotopologues = None,
                 model = "SelfContStandardType",
                 on_the_fly = True,
                 zeeman = False,
                 lineshape = "VP",
                 normalization = "VVH",
                 cutoff = -1,
                 cutoff_type = "ByBand"):
        super().__init__("N2",
                         from_catalog = from_catalog,
                         cia = cia,
                         frequency_range = frequency_range,
                         isotopologues = isotopologues,
                         model = model,
                         on_the_fly = on_the_fly,
                         zeeman = zeeman,
                         lineshape = lineshape,
                         normalization = normalization,
                         cutoff = cutoff,
                         cutoff_type = cutoff_type)

class O2(AbsorptionSpecies):
    def __init__(self,
                 from_catalog = False,
                 cia = None,
                 frequency_range = None,
                 isotopologues = None,
                 model = "PWR93",
                 on_the_fly = True,
                 zeeman = False,
                 lineshape = "VP",
                 normalization = "VVW",
                 cutoff = -1,
                 cutoff_type = "ByBand"):
        super().__init__("O2",
                         from_catalog = from_catalog,
                         cia = cia,
                         frequency_range = frequency_range,
                         isotopologues = isotopologues,
                         model = model,
                         on_the_fly = on_the_fly,
                         zeeman = zeeman,
                         lineshape = lineshape,
                         normalization = normalization,
                         cutoff = cutoff,
                         cutoff_type = cutoff_type)

class CloudWater(AbsorptionSpecies):
    def __init__(self,
                 from_catalog = False,
                 cia = None,
                 frequency_range = None,
                 isotopologues = None,
                 model = "MPM93",
                 on_the_fly = True,
                 zeeman = False,
                 lineshape = "VP",
                 normalization = "RQ",
                 cutoff = -1,
                 cutoff_type = "ByBand"):
        super().__init__("cloud_water",
                         from_catalog = from_catalog,
                         cia = cia,
                         frequency_range = frequency_range,
                         isotopologues = isotopologues,
                         model = model,
                         on_the_fly = on_the_fly,
                         zeeman = zeeman,
                         lineshape = lineshape,
                         normalization = normalization,
                         cutoff = cutoff,
                         cutoff_type = cutoff_type)

    def get_tag_string(self):

        ts = "liquidcloud"
        ts += "-"

        if self._model:
            ts += self._model
            ts += "-"

        return ts
