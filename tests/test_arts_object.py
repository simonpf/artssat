import numpy as np

from parts.arts_object import Dimension, get_shape, broadcast, arts_property, \
    ArtsObject
from parts.arts_object import Dimension as dim
from typhon.arts.workspace import Workspace
from typhon.arts.workspace.variables import workspace_variables as wsv

def test_get_shape():
    """
    Test computation of the shape of nested list.
    """
    s = get_shape(np.ones((5, 5)))
    assert(s == (5, 5))

    s = get_shape([[1, 2]])
    assert(s == (1, 2))

    s = get_shape([1, 2, 3])
    assert(s == (3,))

def test_broadcast():
    """
    Test the broadcasting of nested lists.
    """
    l1 = [[1, 2, 3], [1, 2, 3]]
    l1 = broadcast((2, 3), l1)

    assert(len(l1)    == 2)
    assert(len(l1[0]) == 3)

    l1 = [[1, 2, 3]]
    l1 = broadcast((2, 3), l1)
    assert(len(l1)    == 2)
    assert(len(l1[0]) == 3)

    l1 = [[1], [2]]
    l1 = broadcast((2, 3), l1)
    assert(l1[0] == [1, 1, 1])
    assert(l1[1] == [2, 2, 2])

def test_dimension():
    """
    Test propagation of dimension information.
    """

    dims1 = Dimension()


    n = dims1.infer(dim.P)
    assert(n is None)

    dims1.deduce(dim.P, 10, "temperature grid")

    dims2 = Dimension()
    dims2.link(dims1)
    n, who = dims2.infer(dim.P)
    assert(n == 10)

    dims2.deduce(dim.Lat, 2, "Latitude grid")
    n, who = dims1.infer(dim.Lat)
    assert(n == 2)

def test_dimension_broadcast():
    """
    Test propagation of dimension information for ARTS properties
    as well as broadcasting.
    """

    class A(ArtsObject):

        def __init__(self):
            super().__init__()
            pass

        @arts_property("Vector",
                       shape = (dim.P,),
                       wsv = wsv["p_grid"])
        def p_grid():
            return None

        @arts_property("Tensor3",
                       shape = (dim.P, dim.Lat, dim.Lon),
                       wsv = wsv["t_field"])
        def temperature(self):
            return None

    class TProvider:
        def __init__(self):
            pass

        def get_temperature(self):
            return np.ones((1, 3, 5))

    ws = Workspace()
    a  = A()
    dp = TProvider()

    a.p_grid = np.zeros(4)

    a.setup_arts_properties(ws)
    a.get_data_arts_properties(ws, dp)

    assert(ws.t_field.value.shape[0] == 4)
    assert(np.all(ws.p_grid.value == np.zeros(4)))
