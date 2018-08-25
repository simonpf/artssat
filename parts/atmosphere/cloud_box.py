class CloudBox:
    def __init__(self, n_dimensions, scattering = True):

        self._scattering = scattering
        self._adaptive = None
        self._vertical_limits = None
        self._vertical_limits_type = None
        self._latitude_limits = None
        self._longitude_limits = None

        self._checked = False

    @property
    def limits(self):
        return self._limits

    #
    # Adaptive
    #

    @property
    def adaptive(self):
        return self._adaptive

    @adaptive.setter
    def set_adaptive(self, a):
        self._altitude_limits = None
        self._pressure_limits = None
        self._adaptive = a

    #
    # Altitude limits
    #

    @property
    def altitude_limits(self):
        if self._vertical_limits is None \
           or self._vertical_limits_type != "altitude":
            raise Exception("No altitude limits have been set.")
        return self._vertical_limits

    @altitude_limits.setter
    def set_altitude_limits(self, z1, z2):
        self._adaptive = False
        self._vertical_limits_type = "altitude"

        if not z1 < z2:
            raise Exception("The cloudbox limits pressure limits must "
                            "satisfy z1 > z2.")
        self._vertical_limits = (z1, z2)

    #
    # Pressure limits
    #

    @property
    def pressure_limits(self):
        if self._vertical_limits is None \
           or self._vertical_limits_type != "pressure":
            raise Exception("No pressure limits have been set.")
        return self._pressure_limits

    @pressure_limits.setter
    def set_pressure_limits(self, p1, p2):

        self._adaptive = False
        self._vertical_limits_type == "pressure"

        if not p1 > p2:
            raise Exception("The cloudbox limits pressure limits must "
                            "satisfy p1 > p2.")
        self._pressure_limits = (p1, p2)

    #
    # Latitude limits
    #

    @property
    def latitude_limits(self):
        if self._latitude_limits is None:
            raise Exception("No latitude limits have been set.")
        return self._latituce_limits

    @latitude_limits.setter
    def set_latitude_limits(self, l1, l2):

        if not l1 < l2:
            raise Exception("The latitude limits must satisfy l1 < l2.")

        self._latitude_limits = (l1, l2)

    @property
    def checked(self):
        return self._checked

    def setup(self, ws):
        ws.Copy(ws.iy_cloudbox_agenda, ws.iy_cloudbox_agenda__LinInterpField)

    def get_data(self, ws, provider, *args, **kwargs):

        if not self._scattering:
            ws.jacobianOff()
            ws.cloudboxOff()
            ws.cloudbox_checked = 1
            return

        if not self._adaptive and self._vertical_limits is None:
            ws.cloudboxSetFullAtm()
            return

        if self._adaptive:
            ws.cloudboxSetAutomatically(particle_field =
                                        ws.particle_bulkprop_field)
            return

        v1 = self._vertical_limits[0]
        v2 = self._vertical_limits[1]

        lat_1 = 1.0
        lat_2 = 1.0

        if not self._latitude_limits is None:
            lat_1 = self._latitude_limits[0]
            lat_2 = self._latitude_limits[1]
        elif self._n_dimensions > 1:
            lat_1 = ws.lat_grid.value[0]
            lat_2 = ws.lat_grid.value[-1]

        lon_1 = 1.0
        lon_2 = 1.0

        if not self._longitude_limits is None:
            lon_1 = self._latitude_limits[0]
            lon_2 = self._latitude_limits[1]
        elif self._n_dimensions > 2:
            lon_1 = ws.lon_grid.value[0]
            lon_2 = ws.lon_grid.value[-1]

        if self._vertical_limits_type == "pressure":
            ws.cloudboxSetManually(p1 = v1,
                                   p2 = v2,
                                   lat1 = lat1,
                                   lat2 = lat2,
                                   lon1 = lon1,
                                   lon2 = lon2)

        if self._vertical_limits_type == "altitude":
            ws.cloudboxSetManuallyAltitude(z1 = v1,
                                           z2 = v2,
                                           lat1 = lat1,
                                           lat2 = lat2,
                                           lon1 = lon1,
                                           lon2 = lon2)
            ws._checked = False

    def run_checks(self, ws):

        if not self._scattering:
            return None

        ws.pnd_fieldCalcFromParticleBulkProps()
        ws.cloudbox_checkedCalc()

