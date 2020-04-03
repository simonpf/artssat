"""
The model atmosphere
====================

"""

import numpy as np
from artssat.atmosphere.cloud_box import CloudBox
from artssat.jacobian import  JacobianBase
from artssat.retrieval import RetrievalBase, RetrievalQuantity
from artssat.atmosphere.catalogs import LineCatalog, Perrin

class TemperatureJacobian(JacobianBase):
    def __init__(self,
                 quantity,
                 index,
                 p_grid   = [],
                 lat_grid = [],
                 lon_grid = [],
                 hse      = "on"):
        super().__init__(quantity, index)
        self.p_grid   = p_grid
        self.lat_grid = lat_grid
        self.lon_grid = lon_grid
        self.hse      = hse

    def _make_setup_kwargs(self, ws):

        if self.p_grid.size == 0:
            g1 = ws.p_grid
        else:
            g1 = self.p_grid

        if self.lat_grid.size == 0:
            g2 = ws.lat_grid
        else:
            g2 = self.lat_grid

        if self.lon_grid.size == 0:
            g3 = ws.lon_grid
        else:
            g3 = self.lon_grid

        kwargs = {"g1" : g1, "g2" : g2, "g3" : g3,
                  "hse" : self.hse}

        return kwargs

    def setup(self, ws, data_provider, *args, **kwargs):
        kwargs = self._make_setup_kwargs(ws)
        ws.jacobianAddTemperature(**kwargs)

class TemperatureRetrieval(RetrievalBase, TemperatureJacobian):

    def __init__(self,
                 quantity,
                 index,
                 p_grid   = [],
                 lat_grid = [],
                 lon_grid = [],
                 hse = "on"):
        RetrievalBase.__init__(self)
        TemperatureJacobian.__init__(self, quantity, index,
                                     p_grid, lat_grid, lon_grid, hse)

    def add(self, ws):
        ws.retrievalAddTemperature(**self._make_setup_kwargs(ws))

class Temperature(RetrievalQuantity):


    def __init__(self, atmosphere):
        super().__init__()
        self.atmosphere = atmosphere

    def get_data(self, ws, data_provider, *args, **kwargs):
        t = data_provider.get_temperature(*args, **kwargs)
        self.atmosphere.__check_dimensions__(t, "temperature")
        ws.t_field = self.atmosphere.__reshape__(t)

    def set_from_x(self, ws, xa):
        x = self.transformation.invert(xa)
        self.t_field = x

    @property
    def name(self):
        return "temperature"

    @property
    def jacobian_class(self):
        return TemperatureJacobian

    @property
    def retrieval_class(self):
        return TemperatureRetrieval



class Atmosphere:

    def __init__(self,
                 dimensions,
                 absorbers = [],
                 scatterers = [],
                 surface = None,
                 catalog = None):

        self.__set_dimensions__(dimensions)
        self._required_data = [("p_grid", dimensions[:1], False),
                               ("temperature", dimensions, False),
                               ("altitude", dimensions, False),
                               ("surface_altitude", dimensions[1:], True)]

        self.absorbers  = absorbers
        self.scatterers = scatterers
        self.scattering = len(scatterers) > 0
        self._dimensions = dimensions
        self._cloud_box = CloudBox(n_dimensions = len(dimensions),
                                   scattering = self.scattering)



        self._surface_data_indices = []
        self._surface = surface
        self.temperature = Temperature(self)

        if not surface is None:
            nd = len(self._required_data)
            self._required_data += surface.required_data
            self.surface_data_indices = range(nd, len(self._required_data))

        self._catalog = catalog

    #
    # Dimensions
    #

    def __set_dimensions__(self, dimensions):
        if not type(dimensions) == tuple or not type(dimensions[0]) == int:
            raise Exception("Dimensions of atmosphere must be given as a tuple "
                            "of integers.")
        if not len(dimensions) in [1, 2, 3]:
            raise Exception("The number of dimensions of the atmosphere "
                            "must be 1, 2 or 3.")
        if not all([n >= 0 for n in dimensions]):
            raise Exception("The dimension tuple must contain only  positive "
                            "integers.")
        else:
            self._dimensions = dimensions

    @property
    def dimensions(self):
            return self._dimensions

    #
    # Absorbers
    #

    @property
    def absorbers(self):
        return self._absorbers

    @absorbers.setter
    def absorbers(self, absorbers):
        for a in absorbers:
            self.__dict__[a.name] = a
            self._required_data += [(a.name, self._dimensions, False)]
        self._absorbers = absorbers

    def add_absorber(self, absorber):
        self.__dict__[absorber.name] = absorber
        self._required_data += [(absorber.name, self._dimensions, False)]
        self._absorbers += absorber

    #
    # Cloud box
    #

    @property
    def cloud_box(self):
        return self._cloud_box

    #
    # Catalog
    #

    @property
    def catalog(self):
        """
        Line catalog from which to read absorption line data.
        """
        return self._catalog

    @catalog.setter
    def catalog(self, c):
        if isinstance(c, LineCatalog) or c is None:
            self._catalog = c
        else:
            raise ValueError("Line catalog must be of type LineCatalog.")

    #
    # Jacobian
    #

    def has_jacobian(self):
        for a in self.absorbers:
            if not a.jacobian is None:
                return True
        for b in self.scatterers:
            for m in b.moments:
                if not a.jacobian is None:
                    return True

    #
    # Scatterers
    #

    @property
    def scatterers(self):
        return self._scatterers

    @scatterers.setter
    def scatterers(self, scatterers):

        if not type(scatterers) is list:
            raise ValueError("The 'scatterers' property can only be set to a list.")

        for s in scatterers:
            self.__dict__[s.name] = s
            self._required_data += [(n, self._dimensions, False) \
                                    for n in s.moment_names]
        self._scatterers = scatterers
        self.scattering = True
        self._cloud_box = CloudBox(n_dimensions = len(self.dimensions),
                                   scattering = self.scattering)

    def add_scatterer(self, scatterer):
        self.__dict__[scatterer.name] = scatterer
        self._required_data += [(n, self._dimensions, False) \
                                for n in scatterer.moment_names]
        self._scatterers += [scatterer]
        self.scattering = True
        self._cloud_box = CloudBox(n_dimensions = len(self.dimensions),
                                   scattering = self.scattering)

    #
    # Surface
    #

    @property
    def surface(self):
        return self._surface

    @surface.setter
    def set_surface(self, s):
        if not self._surface is None:
            rd = [d for i, d in enumerate(self._required_data) \
                  if i not in self._required_data_indices]
            nd = len(rd)
            rd += surface.required_data

            self._required_data = rd
            self._requried_data_indices = range(nd, len(rd))
        self._surface = surface

    @property
    def required_data(self):
        return self._required_data

    #
    # Setup
    #

    def __setup_absorption__(self, ws, sensors):
        species = []
        lineshapes = []
        normalizations = []
        cutoffs = []

        for i, a in enumerate(self._absorbers):
            a.setup(ws, i)
            species += [a.get_tag_string()]
        ws.abs_speciesSet(species = species)

        # Set the line shape
        if not self.catalog is None:
            self.catalog.setup(ws, sensors)
            ws.abs_lines_per_speciesCreateFromLines()
            ws.abs_lines_per_speciesSetMirroring(option = "Same")
        else:
            for a in self._absorbers:
                if a.from_catalog:
                    raise Exception("Absorber {} has from_catalog set to true "
                                    "but no catalog is provided".format(a.name))
            ws.abs_lines_per_speciesSetEmpty()

        for i, a in enumerate(self._absorbers):
            tag = a.get_tag_string()
            cutoff = np.float32(a.cutoff)
            cutoff_type = a.cutoff_type
            #ws.abs_lines_per_speciesSetCutoffForSpecies(option = cutoff_type,
            #                                            value = cutoff,
            #                                            species_tag = tag)
            lineshape = a.lineshape
            ws.abs_lines_per_speciesSetLineShapeTypeForSpecies(option = lineshape,
                                                               species_tag = tag)

            normalization = a.normalization
            ws.abs_lines_per_speciesSetNormalizationForSpecies(option = normalization,
                                                               species_tag = tag)



        ws.Copy(ws.abs_xsec_agenda, ws.abs_xsec_agenda__noCIA)
        ws.Copy(ws.propmat_clearsky_agenda,
                ws.propmat_clearsky_agenda__OnTheFly)
        ws.lbl_checkedCalc()

    def __setup_scattering__(self, ws):
        ws.ScatSpeciesInit()
        pb_names = []
        for s in self._scatterers:
            s.setup(ws, len(pb_names))
            pb_names += s.moment_names
        ws.particle_bulkprop_names = pb_names

    def setup(self, ws, sensors):

        if len(self.dimensions) == 1:
            ws.AtmosphereSet1D()
        if len(self.dimensions) == 2:
            ws.AtmosphereSet2D()
        if len(self.dimensions) == 3:
            ws.AtmosphereSet3D()

        self.__setup_absorption__(ws, sensors)
        self.__setup_scattering__(ws)

        self.surface.setup(ws)
        self.cloud_box.setup(ws)

    def setup_jacobian(self, ws):

        for a in self.absorbers:
            a.setup_jacobian(ws)

        for s in self.scatterers:
            for m in s.moments:
                m.setup_jacobian(ws)

    #
    # Data
    #

    def __check_dimensions__(self, f, name):
        s = f.shape

        err = "Provided atmospheric " + name + " field"
        err += " is inconsistent with the dimensions of the atmosphere."

        if len(s) != len(self.dimensions):
            raise Exception(err)
        if not all([i == j or j == 0 for i,j \
                    in zip(s, self.dimensions)]):
            raise Exception(err)

    def __reshape__(self, f):
        s = [1, 1, 1]
        j = 0
        for i in range(len(self.dimensions)):
            if self.dimensions[0] > 0:
                s[i] = self.dimensions[i]
            else:
                s[i] = f.shape[i]
        return np.reshape(f, tuple(s))

    def __get_pressure__(self, ws, provider, *args, **kwargs):
        p = provider.get_pressure(*args, **kwargs).ravel()
        if self.dimensions[0] != 0 and p.size != self.dimensions[0]:
            raise Exception("Provided pressure grid is inconsistent with"
                            " dimensions of the atmosphere.")
        ws.p_grid = p

    def __get_altitude__(self, ws, provider, *args, **kwargs):
        dimensions = ws.t_field.value.shape
        z = provider.get_altitude(*args, **kwargs)
        self.__check_dimensions__(z, "altitude")
        z = self.__reshape__(z)
        if not z.shape == dimensions:
            raise Exception("Dimensions of altitude field inconsistent"
                            " with dimensions of temperature field.")
        ws.z_field = z

        # Surface altitude

        dimensions = ws.t_field.value.shape

        if hasattr(provider, "get_surface_altitude"):
            zs = provider.get_surface_altitude(*args, **kwargs)
            try:
                zs = zs.reshape(dimensions[1:])
                ws.z_surface = zs
            except:
                raise Exception("Shape " + str(zs.shape) + "of provided "
                                "surface altitude is inconsistent with "
                                "the horizontal dimensions of the "
                                "atmosphere " + str(dimensions) + ".")
        else:
            ws.z_surface = ws.z_field.value[0, :, :]

    def __get_absorbers__(self, ws, provider, *args, **kwargs):

        dimensions = ws.t_field.value.shape
        ws.vmr_field = np.zeros((len(self.absorbers),) + dimensions)

        for i, a in enumerate(self.absorbers):
            if a.retrieval is None:
                fname = "get_" + a.name
                f = provider.__getattribute__(fname)
                x = f(*args, **kwargs)
                self.__check_dimensions__(x, a.name)
                x = self.__reshape__(x)

                if not x.shape == dimensions:
                    raise Exception("Dimensions of " + a.name + " VMR field "
                                    "inconcistent with dimensions of temperature "
                                    "field.")
                ws.vmr_field.value[i, :, :, :] = x

        i = 0

        n_moments = sum([len(s.moment_names) for s in self.scatterers])
        ws.particle_bulkprop_field = np.zeros(((n_moments,)
                                               + ws.t_field.value.shape))

    def __get_scatterers__(self, ws, provider, *args, **kwargs):

        if not self.scatterers is None and len(self.scatterers) > 0:
            ws.cloudbox_on = 1
            ws.cloudboxSetFullAtm()

        dimensions = ws.t_field.value.shape

        for s in self.scatterers:
            s.get_data(ws, provider, *args, **kwargs)

    def get_data(self, ws, provider, *args, **kwargs):


        self.__get_pressure__(ws, provider, *args, **kwargs)
        self.temperature.get_data(ws, provider, *args, **kwargs)
        self.__get_altitude__(ws, provider, *args, **kwargs)
        self.__get_absorbers__(ws, provider, *args, **kwargs)
        self.__get_scatterers__(ws, provider, *args, **kwargs)

        self.cloud_box.get_data(ws, provider, *args, **kwargs)
        self.surface.get_data(ws, provider, *args, **kwargs)

    #
    # Checks
    #

    def run_checks(self, ws):

        ws.atmgeom_checkedCalc()
        ws.atmfields_checkedCalc(bad_partition_functions_ok = 1)
        ws.propmat_clearsky_agenda_checkedCalc()
        ws.propmat_clearsky_agenda_checkedCalc()
        ws.abs_xsec_agenda_checkedCalc()

        self.cloud_box.run_checks(ws)
        self.surface.run_checks(ws)


class Atmosphere1D(Atmosphere):

    def __init__(self,
                 absorbers = [],
                 scatterers = [],
                 surface = None,
                 levels = None,
                 catalog = None):
        if levels is None:
            dimensions = (0,)
        else:
            if not type(levels) == int:
                raise Exception("The number of levels of the 1D atmosphere "
                                "must be given by an integer.")
            else:
                dimensions = (level, )
        super().__init__(dimensions,
                         absorbers = absorbers,
                         scatterers = scatterers,
                         surface = surface,
                         catalog = catalog)
