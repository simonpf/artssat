import numpy as np
from parts.atmosphere.atmospheric_quantity \
    import AtmosphericQuantity, extend_dimensions

class Moment(AtmosphericQuantity):
    def __init__(self,
                 species_name,
                 moment_name,
                 jacobian = False):

        name = species_name + "_" + moment_name
        AtmosphericQuantity.__init__(self, (0, 0, 0), jacobian)

        self._species_name = species_name

    #
    # Abstract methods
    #

    def setup(self, ws, i):
        self._wsv_index = i

    def get_data(self, ws, provider, *args, **kwargs):
        AtmosphericQuantity.get_data(self, ws, provider, *args, **kwargs)

    def setup_jacobian(self, ws):
        kwargs = {"species" : self.name,
                  "quantity" : self._species_name}

        if not self.jacobian.p_grid is None:
            kwargs["g1"] = self.jacobian.p_grid

        if not self.jacobian.lat_grid is None:
            kwargs["g2"] = self.jacobian.lat_grid

        if not self.jacobian.lon_grid is None:
            kwargs["g3"] = self.jacobian.lon_grid

        ws.jacobianAddScatSpecies(**kwargs)

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

        if not scattering_data[-4:] == ".xml":
            scattering_data += ".xml"

        self._scattering_data = scattering_data

        if scattering_meta_data is None:
            md = scattering_data[:-3] + "meta.xml"
            self._scattering_meta_data = md

        self._moments = []
        for m in self.moment_names:
            moment = Moment(self.name, m)
            self._moments += [moment]
            self.__dict__[m] = moment

    @property
    def name(self):
        return self._name

    @property
    def moment_names(self):
        name = self._name
        return [name + "_" + m for m in self._psd.moments]

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
        for m in moments:
            m.get_data(ws, i, *args, **kwargs)

    def setup_jacobian(self, ws):
        for m in moments:
            m.setup_jacobian(ws)
