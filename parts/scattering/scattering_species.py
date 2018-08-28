import numpy as np
from parts.atmosphere.atmospheric_quantity \
    import AtmosphericQuantity, extend_dimensions
from parts.arts_object import add_property
import parts.dimensions as dim

class Jacobian:

    def __init__(self, quantity):

        self.quantity = quantity
        self.p_grid   = None
        self.lat_grid = None
        self.lon_grid = None


    def _make_setup_kwargs(self, ws):

        if self.p_grid is None:
            g1 = ws.p_grid
        else:
            g1 = self.p_grid

        if self.lat_grid is None:
            g2 = ws.lat_grid
        else:
            g2 = self.lat_grid

        if self.lon_grid is None:
            g3 = ws.lon_grid
        else:
            g3 = self.lon_grid

        kwargs = {"g1" : g1, "g2" : g2, "g3" : g3,
                  "species" : self.quantity._species_name,
                  "quantity" : self.quantity.name}

        return kwargs

    def setup_jacobian(self, ws):

        kwargs = self._make_setup_kwargs(ws)
        ws.jacobianAddScatSpecies(**kwargs)

class Retrieval(Jacobian):

    def __init__(self, quantity):
        super().__init__(quantity)

        add_property(self, "covariance_matrix", (dim.joker, dim.joker),
                        np.ndarray)
        add_property(self, "xa", (dim.joker), np.ndarray)
        add_property(self, "x0", (dim.joker), np.ndarray)

    def setup_retrieval(self, ws, retrieval_provider, *args, **kwargs):

        fname = "get_" + self.quantity.name + "_covariance"
        covmat_fun = getattr(retrieval_provider, fname)
        covmat = covmat_fun(*args, **kwargs)

        ws.covmat_block = covmat

        ws.retrievalAddScatSpecies(**self._make_setup_kwargs(ws))

        fname = "get_" + self.quantity.name + "_xa"
        xa_fun = getattr(retrieval_provider, fname)
        self.xa = xa_fun(*args, **kwargs)

        fname = "get_" + self.quantity.name + "_x0"
        if hasattr(retrieval_provider, fname):
            x0_fun = getattr(retrieval_provider, fname)
            self.x0 = x0_fun(*args, **kwargs)
        else:
            self.x0 = None

class Moment(AtmosphericQuantity):
    def __init__(self,
                 species_name,
                 moment_name,
                 jacobian = False):

        name = species_name + "_" + moment_name
        super().__init__(name, (0, 0, 0), jacobian)

        self._jacobian = None
        self._retrieval = None
        self._species_name = species_name
        self._moment_name  = moment_name

    #
    # Abstract methods
    #

    def setup(self, ws, i):
        self._wsv = ws.particle_bulkprop_field
        self._wsv_index = i

    def setup_jacobian():
        pass

    def get_data(self, ws, provider, *args, **kwargs):
        AtmosphericQuantity.get_data(self, ws, provider, *args, **kwargs)

    #
    # Jacobian
    #

    @property
    def jacobian(self):
        if self._jacobian is None:
            self._jacobian = Jacobian(self)
        return self._jacobian

    @property
    def retrieval(self):
        if self._retrieval is None:
            self._retrieval = Retrieval(self)
        return self._retrieval

    #
    # Properties
    #

    def species_name(self):
        return self._species_name

class ScatteringSpecies:
    def __init__(self,
                 name,
                 psd,
                 scattering_data,
                 scattering_meta_data = None,
                 jacobian = False):

        self._name = name
        self._psd  = psd

        if not scattering_data[-4:] in [".xml", "l.gz"]:
            scattering_data += ".xml"

        self._scattering_data = scattering_data

        if scattering_meta_data is None:
            md = scattering_data[:-3] + "meta.xml"
            self._scattering_meta_data = md
        else:
            self._scattering_meta_data = scattering_meta_data

        self._moments = []
        for m in self.psd.moment_names:
            moment = Moment(self.name, m)
            self._moments += [moment]
            self.__dict__[m] = moment

    @property
    def name(self):
        return self._name

    @property
    def moment_names(self):
        name = self._name
        return [name + "_" + m for m in self._psd.moment_names]

    @property
    def psd(self):
        return self._psd

    @property
    def scattering_data(self):
        return self._scattering_data

    @property
    def scattering_meta_data(self):
        return self._scattering_meta_data

    @property
    def jacobian(self):
        return self._jacobian

    @property
    def moments(self):
        return self._moments

    @property
    def pnd_agenda(self):
        return self._psd.agenda

    def setup(self, ws, i):

        wsv_name = str(id(self))
        ws.ArrayOfSingleScatteringDataCreate(wsv_name)
        ws.ArrayOfScatteringMetaDataCreate(wsv_name + "_meta")
        scat_data_wsv = ws.__getattr__(wsv_name)
        scat_meta_data_wsv = ws.__getattr__(wsv_name + "_meta")
        ws.ReadXML(scat_data_wsv, self.scattering_data)
        ws.ReadXML(scat_meta_data_wsv, self.scattering_meta_data)

        ws.Append(ws.scat_data_raw, scat_data_wsv)
        ws.Append(ws.scat_meta, scat_meta_data_wsv)

        ws.Delete(scat_data_wsv)
        ws.Delete(scat_meta_data_wsv)

        ws.Append(ws.pnd_agenda_array, self.pnd_agenda)
        ws.Append(ws.scat_species, self.name)
        ws.Append(ws.pnd_agenda_array_input_names, self.moment_names)

        for j, m in enumerate(self.moments):
            m.setup(ws, i + j)

    def get_data(self, ws, provider, *args, **kwargs):
        for m in self.moments:
            m.get_data(ws, provider, *args, **kwargs)

    def setup_jacobian(self, ws):
        for m in moments:
            m.setup_jacobian(ws)
