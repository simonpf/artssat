import numpy as np
import scipy as sp
import scipy.interpolate
from parts.atmosphere.atmospheric_quantity \
    import AtmosphericQuantity, extend_dimensions
from parts.arts_object import add_property

from parts.jacobian  import JacobianBase
from parts.retrieval import RetrievalBase, RetrievalQuantity

from typhon.arts.workspace import arts_agenda

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
        if self.retrieval is None:
            AtmosphericQuantity.get_data(self, ws, provider, *args, **kwargs)

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
        x = self.transformation.invert(x)


        grids = [ws.p_grid.value, ws.lat_grid.value, ws.lon_grid.value]
        grids = [g for g in grids if g.size > 0]
        retrieval_grids = [self.retrieval.p_grid,
                           self.retrieval.lat_grid,
                           self.retrieval.lon_grid]
        used_grids = []
        print(retrieval_grids)
        for i, g in enumerate(grids):
            if retrieval_grids[i].size == 0:
                used_grids += [g]
            else:
                used_grids += [retrieval_grids[i]]
        retrieval_grids = used_grids

        retrieval_grids_shape = [g.size for g in retrieval_grids]
        x = np.reshape(x, retrieval_grids_shape)

        x = x[::-1]
        retrieval_grids[0] = retrieval_grids[0][::-1]
        grids[0] = grids[0][::-1]

        interp = sp.interpolate.RegularGridInterpolator(retrieval_grids, x,
                                                        method = "linear",
                                                        bounds_error = False,
                                                        fill_value = None)

        mesh_grids = np.meshgrid(grids)
        if len(mesh_grids) > 1:
            xi = np.transpose(np.stack(mesh_grids), axes = (0, -1))
        else:
            xi = mesh_grids[0].reshape(-1, 1)
        y  = interp(xi)

        pbf_shape  = [1, 1, 1]
        for i, g in enumerate(grids):
            pbf_shape[i] = g.size
        y = np.reshape(y, pbf_shape)
        y = y[::-1, :, :]

        ws.particle_bulkprop_field.value[self._wsv_index, :, :, :] = np.reshape(y, pbf_shape)

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
