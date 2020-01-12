import numpy as np
import os
from abc import ABCMeta, abstractmethod, abstractproperty

class LineCatalog(metaclass = ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def setup(self, workspace):
        pass

class Hitran(LineCatalog):
    def __init__(self, path = None):

        self.path = path
        if self.path is None:
            try:
                self.path = os.environ["PATH"]
            except:
                self.path = "HITRAN2012.par"

    def setup(self, workspace, sensors):

        f_max = 0.0
        f_min = np.finfo(np.float64).max
        for s in sensors:
            f_min = min(s.f_grid.min(), f_min)
            f_max = max(s.f_grid.max(), f_max)

        workspace.abs_linesReadFromHitran(self.path, f_min, f_max)

class Perrin(LineCatalog):
    def __init__(self, path = None):

        self.path = path
        if self.path is None:
            try:
                self.path = os.environ["PERRIN_PATH"]
            except:
                self.path = "spectroscopy/Perrin/"

    def setup(self, workspace, sensors):

        f_max = 0.0
        f_min = np.finfo(np.float64).max
        for s in sensors:
            f_min = min(s.f_grid.min(), f_min)
            f_max = max(s.f_grid.max(), f_max)

        workspace.ReadSplitARTSCAT(basename = self.path,
                                   fmin = f_min,
                                   fmax = f_max)


class Aer(LineCatalog):
    def __init__(self, path):

        self.path = path

    def setup(self, workspace, sensors):
        try:
            workspace.ReadXML(workspace.abs_lines, self.path)
        except:
            workspace.ReadARTSCAT(filename = self.path)
