from parts.atmosphere.atmospheric_quantity \
    import AtmosphericQuantity, extend_dimensions

class AbsorptionSpecies(AtmosphericQuantity):
    def __init__(self,
                 name,
                 catalog = None,
                 cia = None,
                 frequency_range = None,
                 isotopologues = None,
                 jacobian = False,
                 model = None,
                 on_the_fly = True,
                 zeeman = False):

        AtmosphericQuantity.__init__(self,
                                     name,
                                     (0, 0, 0),
                                     jacobian = jacobian)

        self._dimensions = (0, 0, 0)

        if jacobian:
            self.jacobian.unit = "vmr"
            self.jacobian.for_species_tag = 1

        self._catalog         = catalog
        self._cia             = cia
        self._frequency_range = frequency_range
        self._isotopologues   = isotopologues
        self._model           = model
        self._on_the_fly      = on_the_fly
        self._zeeman          = zeeman

    #
    # Abstract properties
    #

    def dimensions(self):
        return self._dimensions

    #
    # Properties
    #

    @property
    def catalog(self):
        return self._catalog

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

    def get_tag_string(self):

        ts = self._name
        ts += "-"

        if self._zeeman:
            ts += "Z"
            ts += "-"

        if not self._isotopologues is None:
            ts += self._isotopologues
            ts += "-"

        if self._model:
            ts += self._model
            ts += "-"

        if self._frequency_range:
           ts += str(self._frequency_range[0])
           ts += "-"
           ts += str(self._frequency_range[1])

        return ts

    #
    # Abstract methods
    #

    def setup(self, ws, i):
        self._wsv_index = i

    def setup_jacobian(self, ws):
        if not self.jacobian is None:
            kwargs =  {}
            if self.jacobian.perturbation:
                kwargs["dx"] = self.jacobian.perturbation
            if self.jacobian.p_grid:
                kwargs["g1"] = self.jacobian.p_grid
            if self.jacobian.lat_grid:
                kwargs["g2"] = self.jacobian.lat_grid
            if self.jacobian.lon_grid:
                kwargs["g2"] = self.jacobian.lon_grid

            kwargs["unit"] = self.jacobian.unit
            kwargs["for_species_tag"] = self.jacobian.for_species_tag

            ws.jacobianAddAbsSpecies(**kwargs)

    def get_data(self, ws, provider, *args, **kwargs):
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
                 catalog = None,
                 cia = None,
                 frequency_range = None,
                 isotopologues = None,
                 model = None,
                 on_the_fly = True,
                 zeeman = False):
        super().__init__("H2O",
                         catalog = catalog,
                         cia = cia,
                         frequency_range = frequency_range,
                         isotopologues = isotopologues,
                         model = "PWR98",
                         on_the_fly = on_the_fly,
                         zeeman = zeeman)

class N2(AbsorptionSpecies):
    def __init__(self,
                 catalog = None,
                 cia = None,
                 frequency_range = None,
                 isotopologues = None,
                 model = "SelfContStandardType",
                 on_the_fly = True,
                 zeeman = False):
        super().__init__("N2",
                         catalog = catalog,
                         cia = cia,
                         frequency_range = frequency_range,
                         isotopologues = isotopologues,
                         model = model,
                         on_the_fly = on_the_fly,
                         zeeman = zeeman)

class O2(AbsorptionSpecies):
    def __init__(self,
                 catalog = None,
                 cia = None,
                 frequency_range = None,
                 isotopologues = None,
                 model = "PWR93",
                 on_the_fly = True,
                 zeeman = False):
        super().__init__("O2",
                         catalog = catalog,
                         cia = cia,
                         frequency_range = frequency_range,
                         isotopologues = isotopologues,
                         model = model,
                         on_the_fly = on_the_fly,
                         zeeman = zeeman)
