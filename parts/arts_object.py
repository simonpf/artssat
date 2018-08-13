"""The ArtsObject meta class.

The ArtsObject meta class implements functionality common
to the handling of data that is required in an ARTS
simulation. An ARTS object is an object, that holds ARTS
data. To make the handling of these objects easy, they
are exposed as properties of the ArtsObject to the user.
However, they also hold additional information, such as
the expected dimensions and type as well as whether the
value has been fixed by the user or whether it has to
be provided by the data provider.
"""
import numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty

def get_shape(obj):
    """
    Get the shape of a :code:'numpy.ndarray' or of a nested list.

    Parameters(obj):
       obj: The object of which to determine the shape.

    Returns:

        A tuple describing the shape of the :code:`ndarray` or the
        nested list or :code:`(1,)`` if obj is not an instance of either
        of these types.

    """
    if type(obj) == np.ndarray:
        return np.shape(a)
    elif type(obj) == list:
        shape = ()
        inner = obj
        while type(inner[0]) == list:
            shape += (len(inner),)
            inner = inner[0]
        shape += (len(inner),)
        return shape
    else:
        return (1,)

def make_init(properties, old_init):
    """
    Factory function that creates a wrapper for the provided :code:`__init__`
    function that initializes attributes corresponding to the list
    of ARTS properties given in :code:`properties`.

    Parameters:
        properties(list): List containing triplets of property names,
            expected dimension and expected type for the corresponding
            properties.
        old_init(function): The old :code:(__init__) function to be
            wrapped.

    Returns:
        The wrapper around the provided :code:`__init__` function.
    """
    def new_init(self, *args, **kwargs):
        for (p, t, dims) in properties:
            self.__setattr__("_" + p, PlaceHolder(p, t, dims))
        old_init(self, *args, **kwargs)
    return new_init

def make_getter(name):
    """
    Factory function producing a getter for a property called
    :code:`name`.

    The getter function returns the :code:`value`
    attribute of the :code:`PlaceHolder` object that is stored
    as an attribute with name `"_" + name` of the :code:`self`
    object.

    Parameters:

        name(str): The name of the property.

    Returns:

        A getter method for a property with the given name.

    """
    def get(self):
        return getattr(self, "_" + name).value

    return get

def make_setter(name):
    """
    Factory function producing a setter for a property called
    :code:`name`.

    The setter function sets the :code:`value` attribute of the
    PlaceHolder object, which is an attribute of name
    :code:`"_" + name` of the :code:`self` object.

    The setter function checks the type of the provided object
    against the expected type of the property and throws an
    exception if they don't match.

    Parameters:

        name(str): The name of the property.

    Returns:

        A setter method for a property with the given name.

    """
    def set(self, x):
        prop = getattr(self, "_" + name)
        if prop.expected_type == np.ndarray:
            x = np.asarray(x)
        if not prop.expected_type == type(x):
            raise Exception("Type of provided value for {0} doesn't match "
                            "the expected type {1}.".format(name,
                                                            prop.expected_type))
        p = getattr(self, "_" + name)

        p.value = x
        p.fixed = True

    return set

class PlaceHolder:
    """
    Data that is required for an ARTS simulation and can be set
    either to a fixed value, or taken from a data provider.
    """
    def __init__(self, name, dimensions, expected_type):
        """
        Create a PlaceHolder object representing the property
        :code:`name` with expected dimensions :code:`dimensions` and
        expected type :code:`expected_type`.

        Parameters:

            name(str): The name of the property for which this object
                is the placeholder.
            dimensions(tuple): A tuple of :code:`Dimension` objects
                describing the expected dimensions of the corresponding
                property.
            expected_type(type): The expected type for the corresponding
                property.
        """
        self.fixed = False
        self.value = None
        self.expected_type = expected_type
        self.dimensions = dimensions

class ArtsObject(ABCMeta):
    """
    The ArtsObject meta class looks for a class attribute :code:`properties`
    and when found adds getters and setter as well as corresponding
    Placeholder objects as attributes to the class.

    The :code:`properties` attribute is expected to consist of a list
    of triplets containing the name of the property, the expected dimension
    as a tuple of :code:`Dimension` objects, and the expected type for
    this property.

    The ArtsObject meta class ensures that the data for these
    ARTS properties can be set directly through the attributes
    of the inhereting class and hides away the the placeholder
    objects from users of the class.
    """
    def __new__(cls, name, bases, dct):

        if "properties" in dct:
            ps = dct["properties"]

            dct["__init__"] = make_init(ps, dct["__init__"])

            for name, dim, t in ps:
                getter = make_getter(name)

                # Check for custom setter.
                if name + "_setter" in dct:
                    setter = dct[name + "_setter"]
                else:
                    setter = make_setter(name)

                prop = property(getter, setter, name)
                dct[name] = prop

        return super().__new__(cls, name, bases, dct)
