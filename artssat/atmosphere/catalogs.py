"""
artssat.atmosphere.catalogs
===========================

Class to represent absorption line catalogs.
"""
import numpy as np
import os


class LineCatalog:
    """
    Interface to read lines from a generic line catalog.
    """

    def __init__(self, path):
        self.path = path

    def setup(self, workspace, sensors):
        workspace.ReadXML(workspace.abs_lines, self.path)


class Hitran(LineCatalog):
    def __init__(self, path=None):

        self.path = path
        if self.path is None:
            try:
                self.path = os.environ["HITRAN_PATH"]
            except:
                self.path = "HITRAN2012.par"

    def setup(self, workspace, sensors):
        f_max = 0.0
        f_min = np.finfo(np.float64).max
        for s in sensors:
            f_min = min(s.f_grid.min(), f_min)
            f_max = max(s.f_grid.max(), f_max)
        workspace.abs_lines_per_speciesReadSpeciesSplitCatalog(basename=self.path)


class Perrin(LineCatalog):
    """
    Interface for reading Perrin catalog.
    """

    def __init__(self, path=None):

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

        df = f_max - f_min
        f_min = np.maximum(f_min - 0.5 * df, 0.0)
        f_max = f_max + 0.5 * df

        workspace.ReadSplitARTSCAT(basename=self.path, fmin=f_min, fmax=f_max)


class Aer(LineCatalog):
    """
    Interface for reading AER catalog.
    """

    def __init__(self, path):

        self.path = path

    def setup(self, workspace, sensors):
        try:
            workspace.ReadXML(workspace.abs_lines, self.path)
        except:
            workspace.ReadARTSCAT(filename=self.path)
