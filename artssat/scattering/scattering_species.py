import numpy as np
from artssat.atmosphere.atmospheric_quantity \
    import AtmosphericQuantity, extend_dimensions
from artssat.arts_object import add_property
from artssat.scattering.psd.arts.arts_psd import ArtsPSD

from artssat.jacobian  import JacobianBase
from artssat.retrieval import RetrievalBase, RetrievalQuantity

from pyarts.workspace import arts_agenda

################################################################################
# Jacobian
################################################################################

class Jacobian(JacobianBase):

    def __init__(self, quantity, index):
        super().__init__(quantity, index)


    def _make_setup_kwargs(self, ws):

        kwargs = self.get_grids(ws)
        kwargs.update({"species" : self.quantity._species_name,
                      "quantity" : self.quantity.name})

        return kwargs

    def setup(self, ws):

        kwargs = self._make_setup_kwargs(ws)
        ws.jacobianAddScatSpecies(**kwargs)

################################################################################
# Retrieval
################################################################################

class Retrieval(RetrievalBase, Jacobian):

    def __init__(self, quantity, index):
        RetrievalBase.__init__(self)
        Jacobian.__init__(self, quantity, index)

    def add(self, ws):
        ws.retrievalAddScatSpecies(**self._make_setup_kwargs(ws))

class Moment(AtmosphericQuantity, RetrievalQuantity):
    def __init__(self,
                 species_name,
                 moment_name,
                 data = None,
                 jacobian = False):

        name = species_name + "_" + moment_name
        AtmosphericQuantity.__init__(self, name, (0, 0, 0), jacobian)
        RetrievalQuantity.__init__(self)


        self._jacobian = None
        self._retrieval = None
        self._species_name = species_name
        self._moment_name  = moment_name
        self._data = data

    #
    # Abstract methods
    #

    def setup(self, ws, i):
        self._wsv = ws.particle_bulkprop_field
        self._wsv_index = i

    def setup_jacobian():
        pass

    def get_data(self, ws, provider, *args, **kwargs):

        if self.retrieval is None:

            if hasattr(provider, "get_" + self.name):
                AtmosphericQuantity.get_data(self, ws, provider, *args, **kwargs)
            else:
                try:
                    x = self.data
                    pbf_shape = ws.particle_bulkprop_field.value.shape[1:]
                    if len(x.shape) < len(pbf_shape):
                        n = len(pbf_shape) - len(x.shape)
                        x = np.reshape(x, x.shape + (1,) * n)
                    ws.particle_bulkprop_field.value[self._wsv_index, :, :, :] \
                        = np.broadcast_to(x, pbf_shape)
                except Exception as e:
                    raise Exception("Encountered error trying to get data for "
                                    " moment {0}: {1}".format(self.name, e))

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

        if not self.retrieval.limit_high == None:
            x = np.minimum(x, self.retrieval.limit_high)

        if not self.retrieval.limit_low == None:
            x = np.maximum(x, self.retrieval.limit_low)

        grids = [ws.p_grid.value, ws.lat_grid.value, ws.lon_grid.value]
        grids = [g for g in grids if g.size > 0]

        x = self.transformation.invert(x)
        x = self.retrieval.interpolate_to_grids(x, grids)

        pbf_shape = ws.particle_bulkprop_field.value.shape[1:]
        ws.particle_bulkprop_field.value[self._wsv_index, :, :, :] = np.reshape(x, pbf_shape)

    #
    # Properties
    #

    @property
    def data(self):
        return self._data

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

        if hasattr(scattering_data, "path") and hasattr(scattering_data, "meta"):
            scattering_meta_data = scattering_data.meta
            scattering_data = scattering_data.path

        if not scattering_data[-4:] in [".xml", "l.gz"]:
            scattering_data += ".xml"

        self._scattering_data = scattering_data

        if scattering_meta_data is None:
            md = scattering_data[:-3] + "meta.xml"
            self._scattering_meta_data = md
        else:
            self._scattering_meta_data = scattering_meta_data

        self._moments = []

        try:
            moment_data = self.psd.moments
            for m, d in zip(self.psd.moment_names, moment_data):
                moment = Moment(self.name, m, data = d)
                self._moments += [moment]
                self.__dict__[m] = moment
        except:
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

    @psd.setter
    def psd(self, psd):
        if not isinstance(psd, ArtsPSD):
            raise ValueError("PSD of scattering species must implement the ArtsPsd ABC.")
        self._psd = psd

    @property
    def scattering_data(self):
        return self._scattering_data

    @scattering_data.setter
    def scattering_data(self, scattering_data):

        if type(scattering_data) == tuple:
            scattering_data, scattering_meta_data = scattering_data
        elif  hasattr(scattering_data, "path") and hasattr(scattering_data, "meta"):
            scattering_meta_data = scattering_data.meta
            scattering_data = scattering_data.path
        else:
            scattering_meta_data = None

        if not scattering_data[-4:] in [".xml", "l.gz"]:
            scattering_data += ".xml"

        self._scattering_data = scattering_data

        if scattering_meta_data is None:
            md = scattering_data[:-3] + "meta.xml"
            self._scattering_meta_data = md
        else:
            self._scattering_meta_data = scattering_meta_data

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

        if hasattr(self.psd, "setup"):
            self.psd.setup(ws, i)

    def get_data(self, ws, provider, *args, **kwargs):
        for m in self.moments:
            m.get_data(ws, provider, *args, **kwargs)

        if hasattr(self.psd, "get_data"):
            self.psd.get_data(ws, provider, *args, **kwargs)

    def setup_jacobian(self, ws):
        for m in moments:
            m.setup_jacobian(ws)
