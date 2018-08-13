import numpy as np
from parts.atmosphere.scattering import CloudBox

class Atmosphere:

    def __init__(self,
                 dimensions,
                 absorbers = [],
                 scatterers = [],
                 surface = None):

        self.__set_dimensions__(dimensions)
        self._required_data = [("p_grid", dimensions[:1], False),
                               ("temperature", dimensions, False),
                               ("altitude", dimensions, False),
                               ("surface_altitude", dimensions[1:], True)]

        self.absorbers  = absorbers
        self.scatterers = scatterers
        self._dimensions = dimensions
        self._cloud_box = CloudBox(n_dimensions = len(dimensions))



        self._surface_data_indices = []
        self._surface = surface

        if not surface is None:
            nd = len(self._required_data)
            self._required_data += surface.required_data
            self.surface_data_indices = range(nd, len(self._required_data))

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
        for s in scatterers:
            self.__dict__[s.name] = s
            self._required_data += [(n, self._dimensions, False) \
                                    for n in s.moment_names]
        self._scatterers = scatterers

    def add_scatterer(self, scatterer):
        self.__dict__[scatterer.name] = scatterer
        self._required_data += [(n, self._dimensions, False) \
                                for n in s.moment_names]
        self._absorbers += absorber

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

    def __setup_absorption__(self, ws):
        species = []
        for a in self._absorbers:
            species += [a.get_tag_string()]

        ws.abs_speciesSet(species = species)
        ws.abs_lines_per_speciesSetEmpty()
        ws.Copy(ws.abs_xsec_agenda, ws.abs_xsec_agenda__noCIA)
        ws.Copy(ws.propmat_clearsky_agenda,
                ws.propmat_clearsky_agenda__OnTheFly)

    def __setup_scattering__(self, ws):

        ws.ScatSpeciesInit()
        pb_names = []
        for s in self._scatterers:
            s.setup(ws, len(pb_names))
            pb_names += s.moment_names
        ws.particle_bulkprop_names = pb_names

    def setup(self, ws):

        if len(self.dimensions) == 1:
            ws.AtmosphereSet1D()
        if len(self.dimensions) == 2:
            ws.AtmosphereSet2D()
        if len(self.dimensions) == 3:
            ws.AtmosphereSet3D()

        self.__setup_absorption__(ws)
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

    def __get_temperature__(self, ws, provider, *args, **kwargs):
        t = provider.get_temperature(*args, **kwargs)
        self.__check_dimensions__(t, "temperature")
        ws.t_field = self.__reshape__(t)

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
            zs = provider.get_surface_altitude()
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

        i = 0
        for s in self.scatterers:
            fname = "get_" + s.name
            f = provider.__getattribute__(fname)
            x = f(s.psd, *args, **kwargs)

            if not len(x) == len(s.moment_names):
                raise Exception("Bulk property data provided for scattering "
                                "species " + s.name + " is inconsistent "
                                "with PSD.")

            for j,m in enumerate(x):

                print(m)
                self.__check_dimensions__(m, s.moment_names[j])
                m = self.__reshape__(m)

                if not m.shape == dimensions:
                    raise Exception("Dimensions of " + s.name + " " +
                                    s.moment_names[i] + " field "
                                    "inconcistent with dimensions of "
                                    " temperature field.")
                ws.particle_bulkprop_field.value[i, :, :, :] = m
                i += 1



    def get_data(self, ws, provider, *args, **kwargs):

        self.__get_pressure__(ws, provider, *args, **kwargs)
        self.__get_temperature__(ws, provider, *args, **kwargs)
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
        ws.atmfields_checkedCalc()
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
                 levels = None):
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
                         surface = surface)
