from parts.data_provider import DataProviderBase
import numpy as np
import scipy as sp
import scipy.sparse

class DataProviderAPriori(DataProviderBase):

    def __init__(self,
                 name,
                 covariance,
                 spatial_correlation = None):

        super().__init__()
        xa_name = "get_" + name + "_xa"
        self.__dict__[xa_name] = self.get_xa
        covariance_name = "get_" + name + "_covariance"
        self.__dict__[covariance_name] = self.get_covariance

        self.name = name
        self.covariance = covariance
        self.spatial_correlation = spatial_correlation

    def get_xa(self, *args, **kwargs):

        f_name = "get_" + self.name
        try:
            f = getattr(self.owner, f_name)
        except:
            raise Expetion("DataProviderApriori instance requires get method "
                           " {0} from its owning data provider.")

        x = f(*args, **kwargs)
        return x

    def get_covariance(self, *args, **kwargs):

        z = self.owner.get_altitude(*args, **kwargs)
        t = self.owner.get_temperature(*args, **kwargs)

        if (np.array(self.covariance).size == 1):
            diag = self.covariance * np.ones(t.shape)
        else:
            diag = self.covariance

        covmat = np.diag(diag)

        if not self.spatial_correlation is None:
            covmat = self.spatial_correlation(covmat, z)

        print(covmat.shape)

        return covmat


