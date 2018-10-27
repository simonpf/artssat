from parts.scattering.psd.arts.arts_psd import ArtsPSD
from parts.scattering.psd.data.psd_data import PSDData, D_eq
from parts.arts_object import arts_property
from parts.arts_object import Dimension as dim
from parts.arts_object import ArtsObjectReplacement
from typhon.arts.workspace import Workspace, arts_agenda

ws = Workspace()

class FixedShape(ArtsPSD, ArtsObjectReplacement):

    _wsvs = [ws.create_variable("Vector", "x"),
             ws.create_variable("Matrix", "data")]

    private_wsvs = ["x", "data"]

    @arts_property("Vector")
    def x(self):
        return None

    @arts_property("Matrix")
    def shape(self):
        return None

    def __init__(self, x, data, size_parameter = D_eq(1000.0)):

        ArtsObjectReplacement.__init__(self)
        self.psd = PSDData(x, data, size_parameter)
        ArtsPSD.__init__(self, self.psd.size_parameter)

        shape = self.psd.data.reshape((1, -1))
        self.shape = shape / self.psd.get_mass_density()
        self.size_parameter = self.psd.size_parameter

        self.name = "fixed_psd"
        self._create_private_wsvs(ws, self.private_wsvs)

    @property
    def moment_names(self):
        return ["mass_density"]

    @property
    def pnd_call_agenda(self):
        @arts_agenda
        def pnd_call(ws):
            ws.Ignore(ws.pnd_agenda_input_t)
            ws.Ignore(ws.pnd_agenda_input)
            ws.Ignore(ws.pnd_agenda_input_names)
            ws.Ignore(ws.dpnd_data_dx_names)
            ws.Copy(ws.psd_size_grid, self._wsvs["x"])
            ws.Copy(ws.pnd_size_grid, self._wsvs["x"])
            ws.Copy(ws.psd_data, self._wsvs["data"])
            ws.Print(self._wsvs["data"])
            ws.Touch(ws.dpsd_data_dx)

        return pnd_call

    def setup(self, ws, i):
        self.pbf_index = i
        ws.Copy(self._wsvs["x"], self.psd.x)

    def get_data(self, ws, i, *args, **kwargs):
        md = ws.particle_bulkprop_field.value[self.pbf_index, :, :, :]\
                                       .reshape(-1, 1)
        print((self.shape * md).shape)
        ws.Copy(self._wsvs["data"], self.shape * md)
