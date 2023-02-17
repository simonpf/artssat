"""
artssat.atmosphere.atmosphere
=============================

This module defines the Atmosphere class and related objects.
"""
import numpy as np
from artssat.atmosphere.cloud_box import CloudBox
from artssat.jacobian import JacobianBase
from artssat.retrieval import RetrievalBase, RetrievalQuantity
from artssat.atmosphere.catalogs import LineCatalog, Perrin


class TemperatureJacobian(JacobianBase):
    """
    Represents a Jacobian calculation for the atmospheric temperature.
    """

    def __init__(self, quantity, index, p_grid=[], lat_grid=[], lon_grid=[], hse="on"):
        """ """
        super().__init__(quantity, index)
        self.p_grid = p_grid
        self.lat_grid = lat_grid
        self.lon_grid = lon_grid
        self.hse = hse

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

        kwargs = {"g1": g1, "g2": g2, "g3": g3, "hse": self.hse}

        return kwargs

    def setup(self, ws, data_provider, *args, **kwargs):
        kwargs = self._make_setup_kwargs(ws)
        ws.jacobianAddTemperature(**kwargs)


class TemperatureRetrieval(RetrievalBase, TemperatureJacobian):
    def __init__(self, quantity, index, p_grid=[], lat_grid=[], lon_grid=[], hse="on"):
        RetrievalBase.__init__(self)
        TemperatureJacobian.__init__(
            self, quantity, index, p_grid, lat_grid, lon_grid, hse
        )

    def add(self, ws):
        ws.retrievalAddTemperature(**self._make_setup_kwargs(ws))


class Temperature(RetrievalQuantity):
    def __init__(self, atmosphere):
        super().__init__()
        self.atmosphere = atmosphere

    def get_data(self, ws, data_provider, *args, **kwargs):
        t = data_provider.get_temperature(*args, **kwargs)
        self.atmosphere._check_dimensions(t, "temperature")
        ws.t_field = self.atmosphere._reshape(t)

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
    """
    The Atmosphere in which the ARTS simulation takes place.

    The atmosphere comprises a surface as well as several absorbing
    and scattering species.
    """

    def __init__(
        self, dimensions, absorbers=[], scatterers=[], surface=None, catalog=None
    ):
        """
        Args:
            dimensions: The dimensions of the atmosphere
            absorbers: A list containing the absorbing species in the atmosphere.
            scatterers: A list containing the scattering species in the
                atmosphere.
            catalog: The catalog to load the absorption lines from.
        """
        self._set_dimensions(dimensions)
        self._required_data = [
            ("p_grid", dimensions[:1], False),
            ("temperature", dimensions, False),
            ("altitude", dimensions, False),
            ("surface_altitude", dimensions[1:], True),
        ]

        self.absorbers = absorbers
        self.scatterers = scatterers
        self.scattering = len(scatterers) > 0
        self._dimensions = dimensions
        self._cloud_box = CloudBox(
            n_dimensions=len(dimensions), scattering=self.scattering
        )

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

    def _set_dimensions(self, dimensions):
        if not type(dimensions) == tuple or not type(dimensions[0]) == int:
            raise Exception(
                "Dimensions of atmosphere must be given as a tuple " "of integers."
            )
        if not len(dimensions) in [1, 2, 3]:
            raise Exception(
                "The number of dimensions of the atmosphere " "must be 1, 2 or 3."
            )
        if not all([n >= 0 for n in dimensions]):
            raise Exception(
                "The dimension tuple must contain only  positive " "integers."
            )
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
            self._required_data += [
                (n, self._dimensions, False) for n in s.moment_names
            ]
        self._scatterers = scatterers
        self.scattering = True
        self._cloud_box = CloudBox(
            n_dimensions=len(self.dimensions), scattering=self.scattering
        )

    def add_scatterer(self, scatterer):
        self.__dict__[scatterer.name] = scatterer
        self._required_data += [
            (n, self._dimensions, False) for n in scatterer.moment_names
        ]
        self._scatterers += [scatterer]
        self.scattering = True
        self._cloud_box = CloudBox(
            n_dimensions=len(self.dimensions), scattering=self.scattering
        )

    #
    # Surface
    #

    @property
    def surface(self):
        return self._surface

    @surface.setter
    def set_surface(self, surface):
        """
        Set the surface of the atmosphere.
        """
        if not self._surface is None:
            rd = [
                d
                for i, d in enumerate(self._required_data)
                if i not in self._required_data_indices
            ]
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

    def _setup_absorption(self, ws, sensors):
        """
        Sets up the absorption in the atmosphere.

        Args:
            ws: The workspace on which the simulation is performed.
            sensors: The list of sensors included in the simulation.
        """
        species = []
        lineshapes = []
        normalizations = []
        cutoffs = []

        for i, a in enumerate(self._absorbers):
            a.setup(ws, i)
            species += [a.get_tag_string()]
        ws.abs_speciesSet(species=species)

        # Set the line shape
        if not self.catalog is None:
            self.catalog.setup(ws, sensors)
            ws.abs_lines_per_speciesCreateFromLines()
            ws.abs_lines_per_speciesSetMirroring(option="Same")
        else:
            for a in self._absorbers:
                if a.from_catalog:
                    raise Exception(
                        "Absorber {} has from_catalog set to true "
                        "but no catalog is provided".format(a.name)
                    )
            ws.abs_lines_per_speciesSetEmpty()

        for i, a in enumerate(self._absorbers):
            tag = a.get_tag_string()
            cutoff = np.float32(a.cutoff)
            cutoff_type = a.cutoff_type
            lineshape = a.lineshape
            ws.abs_lines_per_speciesSetLineShapeTypeForSpecies(
                option=lineshape, species_tag=tag
            )

            normalization = a.normalization
            ws.abs_lines_per_speciesSetNormalizationForSpecies(
                option=normalization, species_tag=tag
            )

        ws.abs_lines_per_speciesSetCutoffForSpecies(
            option="ByLine", value=100e9, species_tag=tag
        )
        ws.Copy(ws.abs_xsec_agenda, ws.abs_xsec_agenda__noCIA)
        ws.Copy(ws.propmat_clearsky_agenda, ws.propmat_clearsky_agenda__OnTheFly)
        ws.lbl_checkedCalc()

    def _setup_scattering(self, ws):
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

        self._setup_absorption(ws, sensors)
        self._setup_scattering(ws)

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

    def _check_dimensions(self, f, name):
        s = f.shape

        err = "Provided atmospheric " + name + " field"
        err += " is inconsistent with the dimensions of the atmosphere."

        if len(s) != len(self.dimensions):
            raise Exception(err)
        if not all([i == j or j == 0 for i, j in zip(s, self.dimensions)]):
            raise Exception(err)

    def _reshape(self, field):
        """
        Reshape a field to the shape of the atmsophere.

        Args:
            field: The field to reshape.

        Return:
            The given field reshaped to the inferred shape of the
            atmosphere.
        """
        shape = [1, 1, 1]
        for dim in range(len(self.dimensions)):
            if self.dimensions[0] > 0:
                shape[dim] = self.dimensions[dim]
            else:
                shape[dim] = field.shape[dim]
        return np.reshape(field, tuple(shape))

    def _get_pressure(self, ws, provider, *args, **kwargs):
        """
        Get pressure from the data provider and set the ``p_grid`` of
        the ARTS workspace.

        Args:
            ws: The pyarts Workspace on which the simulation is to be performed.
            profider: The data provider.
            *args: The arguments passed to the data provider.
            **kwargs: The key-word arguments passed to the data provider.
        """
        p = provider.get_pressure(*args, **kwargs).ravel()
        if self.dimensions[0] != 0 and p.size != self.dimensions[0]:
            raise Exception(
                "Provided pressure grid is inconsistent with"
                " dimensions of the atmosphere."
            )
        ws.p_grid = p

    def _get_altitude(self, ws, provider, *args, **kwargs):
        """
        Get altitude from the data provider and set the ``z_grid`` of
        the ARTS workspace.

        Args:
            ws: The pyarts Workspace on which the simulation is to be performed.
            profider: The data provider.
            *args: The arguments passed to the data provider.
            **kwargs: The key-word arguments passed to the data provider.
        """
        dimensions = ws.t_field.value.shape
        z = provider.get_altitude(*args, **kwargs)
        self._check_dimensions(z, "altitude")
        z = self._reshape(z)
        if not z.shape == dimensions:
            raise Exception(
                "Dimensions of altitude field inconsistent"
                " with dimensions of temperature field."
            )
        ws.z_field = z

        # Surface altitude
        dimensions = ws.t_field.value.shape

        if hasattr(provider, "get_surface_altitude"):
            zs = provider.get_surface_altitude(*args, **kwargs)
            try:
                zs = zs.reshape(dimensions[1:])
                ws.z_surface = zs
            except:
                raise Exception(
                    "Shape " + str(zs.shape) + "of provided "
                    "surface altitude is inconsistent with "
                    "the horizontal dimensions of the "
                    "atmosphere " + str(dimensions) + "."
                )
        else:
            ws.z_surface = ws.z_field.value[0, :, :]

    def _get_latitude(self, ws, provider, *args, **kwargs):
        if len(self.dimensions) > 1:
            dimensions = ws.t_field.value.shape
            lats = provider.get_latitude(*args, **kwargs)
            ws.lat_grid = np.arange(lats.size)
            ws.lat_true = lats

    def _get_longitude(self, ws, provider, *args, **kwargs):
        if len(self.dimensions) > 1:
            dimensions = ws.t_field.value.shape
            lons = provider.get_longitude(*args, **kwargs)
            ws.lon_true = lons
        if len(self.dimensions) < 3:
            ws.lon_grid = []

    def _get_absorbers(self, ws, provider, *args, **kwargs):

        dimensions = ws.t_field.value.shape
        ws.vmr_field = np.zeros((len(self.absorbers),) + dimensions)

        for i, a in enumerate(self.absorbers):
            if a.retrieval is None:
                fname = "get_" + a.name
                f = provider.__getattribute__(fname)
                x = f(*args, **kwargs)
                self._check_dimensions(x, a.name)
                x = self._reshape(x)

                if not x.shape == dimensions:
                    raise Exception(
                        "Dimensions of " + a.name + " VMR field "
                        "inconsistent with dimensions of temperature "
                        "field."
                    )
                ws.vmr_field.value[i, :, :, :] = x

        i = 0

        n_moments = sum([len(s.moment_names) for s in self.scatterers])
        ws.particle_bulkprop_field = np.zeros(((n_moments,) + ws.t_field.value.shape))

    def _get_scatterers(self, ws, provider, *args, **kwargs):

        dimensions = ws.t_field.value.shape
        ws.cloudbox_on = 1
        ws.cloudbox_limits = [
            0,
            dimensions[0] - 1,
            0,
            dimensions[1] - 1,
            0,
            dimensions[2] - 1,
        ]
        # if not self.scatterers is None and len(self.scatterers) > 0:
        #    ws.cloudboxSetFullAtm()

        for s in self.scatterers:
            s.get_data(ws, provider, *args, **kwargs)

    def get_data(self, ws, provider, *args, **kwargs):

        self._get_pressure(ws, provider, *args, **kwargs)
        self.temperature.get_data(ws, provider, *args, **kwargs)
        self._get_altitude(ws, provider, *args, **kwargs)
        self._get_latitude(ws, provider, *args, **kwargs)
        self._get_longitude(ws, provider, *args, **kwargs)
        self._get_absorbers(ws, provider, *args, **kwargs)
        self._get_scatterers(ws, provider, *args, **kwargs)

        self.cloud_box.get_data(ws, provider, *args, **kwargs)
        self.surface.get_data(ws, provider, *args, **kwargs)

    #
    # Checks
    #

    def run_checks(self, ws):

        ws.atmgeom_checkedCalc()
        ws.atmfields_checkedCalc(bad_partition_functions_ok=1)
        ws.propmat_clearsky_agenda_checkedCalc()
        ws.propmat_clearsky_agenda_checkedCalc()
        ws.abs_xsec_agenda_checkedCalc()

        self.cloud_box.run_checks(ws)
        self.surface.run_checks(ws)


class Atmosphere1D(Atmosphere):
    def __init__(
        self, absorbers=[], scatterers=[], surface=None, levels=None, catalog=None
    ):
        if levels is None:
            dimensions = (0,)
        else:
            if not type(levels) == int:
                raise Exception(
                    "The number of levels of the 1D atmosphere "
                    "must be given by an integer."
                )
            else:
                dimensions = (level,)
        super().__init__(
            dimensions,
            absorbers=absorbers,
            scatterers=scatterers,
            surface=surface,
            catalog=catalog,
        )


class Atmosphere2D(Atmosphere):
    def __init__(
        self, absorbers=[], scatterers=[], surface=None, levels=None, catalog=None
    ):
        if levels is None:
            dimensions = (0, 0)
        super().__init__(
            dimensions,
            absorbers=absorbers,
            scatterers=scatterers,
            surface=surface,
            catalog=catalog,
        )
