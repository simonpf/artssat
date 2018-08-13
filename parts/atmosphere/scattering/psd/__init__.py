import numpy as np
from typhon.arts.workspace import arts_agenda
from scipy.special import gamma

class D14:

    dm_min = 1e-6

    def __init__(self,
                 moments = ["mass_density",
                            "volume_weighted_diameter"],
                 alpha = -0.237,
                 beta  = 1.839,
                 rho = 917.0,
                 size_parameter = "dveq",
                 t_min = 0.0,
                 t_max = 999.0):
        self._alpha   = alpha
        self._beta    = beta
        self._moments = moments
        self._rho = rho
        self._size_parameter = size_parameter
        self._t_min = t_min
        self._t_max = t_max

    def dm_from_md_nd(self, md, nd):
        a = gamma((self.alpha + 5) / self.beta)
        b = gamma((self.alpha + 4) / self.beta)

        t = gamma((self.alpha + 1) / self.beta)
        t *= 6.0 * md * a ** 3.0
        t /= np.pi * self.rho * b ** 4 * nd

        dm = t ** (1.0 / 3.0)
        return np.nan_to_num(dm)

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    @property
    def moments(self):
        return self._moments

    @property
    def rho(self):
        return self._rho
    @property
    def size_parameter(self):
        return self._size_parameter

    @property
    def t_min(self):
        return self._t_min

    @property
    def t_max(self):
        return self._t_max

    @property
    def agenda(self):

        if "intercept_parameter" in self._moments:
            n0Star = np.float("nan")
        else:
            n0Star = -999.0

        if "volume_weighted_diameter" in self._moments:
            dm = np.float("nan")
        else:
            dm = -999.0

        if "mass_density" in self._moments:
            md = np.float("nan")
        else:
            md = -999.0

        @arts_agenda
        def pnd_agenda(ws):
            ws.ScatSpeciesSizeMassInfo(species_index = ws.agenda_array_index,
                                       x_unit = self._size_parameter,
                                       x_fit_start = 100e-6)
            ws.Copy(ws.psd_size_grid, ws.scat_species_x)
            ws.Copy(ws.pnd_size_grid, ws.scat_species_x)
            ws.psdD14(n0Star = n0Star,
                      Dm = dm,
                      iwc = md,
                      rho = self._rho,
                      alpha = self._alpha,
                      beta = self._beta,
                      t_min = self._t_min,
                      Dm_min = D14.dm_min,
                      t_max = self._t_max)
            ws.pndFromPsdBasic()

        return pnd_agenda

    def convert(self, moments, *args):
        if not len(moments) == len(self.moments):
            raise Exception("Need at least as many quantities as moments "
                            "of the PSD ({0})".format(len(self.moments)))
        if moments == self.moments:
            return args

        if "mass_density" in moments and "number_density" in moments:
            i_md = moments.index("mass_density")
            i_nd = moments.index("number_density")
            md = args[i_md]
            nd = args[i_nd]

            a = gamma((self.alpha + 5) / self.beta)
            b = gamma((self.alpha + 4) / self.beta)

            t = gamma((self.alpha + 1) / self.beta)
            t *= 6.0 * md * a ** 3.0
            t /= np.pi * self.rho * b ** 4 * nd

            dm = t ** (1.0 / 3.0)
            dm = np.nan_to_num(dm)
            dm = np.minimum(dm, 1.0)
            dm = np.maximum(dm, D14.dm_min)

            c = gamma(4.0) / 4.0 ** 4
            n0 = 6.0 * md / (np.pi * self.rho * dm ** 4.0)
            n0 = np.nan_to_num(n0)

            i_md = -1
            i_dm = -1
            i_n0 = -1

            if "mass_density" in self.moments:
                i_md = self.moments.index("mass_density")

            if "volume_weighted_diameter" in self.moments:
                i_dm = self.moments.index("volume_weighted_diameter")

            if "intercept_parameter" in self.moments:
                i_n0 = self.moments.index("intercept_parameter")

            results = 2 * [[]]
            if i_md >= 0:
                results[i_md] = md
            if i_dm >= 0:
                results[i_dm] = dm
            if i_n0 >= 0:
                results[i_n0] = n0

            return results
        else:
            raise Exception("Currently only conversion from mass and"
            "number density is supported.")


