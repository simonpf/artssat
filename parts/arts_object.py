"""
parts.arts_object
-----------------

The :code:`arts_object` module provides meta-programming functionality
that aims to simplify unified treatment of relevant ARTS variables.
The basic idea is to provide so called *ARTS properties*, which are
similar in concept to Python properties but extend their functionality to
simplify handling of variables that should be passed to the ARTS workspace.

Concept
=======

From the user perspective, the purpose of ARTS properties is to provide
a simple interface to set ARTS parameters using attributes of Python
objects, hiding away boilerplate code related to the management of
these parameters. From the developer perspective they provide a unified
way to treat these variables as well as default methods for the
interaction with the ARTS workspace. The implementation of ARTS properties
is based on the
`Descriptor <https://docs.python.org/3.7/howto/descriptor.html>`_ protocol.

Usecase
~~~~~~~

The typical usage scenario is that a Python class is used to represent
a conceptual unit grouping together various ARTS functionality and
variables. An example of this is for example the :class:`Sensor` class.
In ARTS, the sensor is described through a set of WSVs. ARTS properties
can then for example be used to expose the :code:`f_grid` WSV as an
attributes of the :code:`Sensor` class.

::

    class Sensor:
        @arts_property
        def f_grid(group = "Vector", shape = (Dimension.Frq), wsv = f_grid):
             return None # No default value

User perspective
~~~~~~~~~~~~~~~~

Seen from the user perspective, ARTS properties provide the following
functionality:

1. Provision of default values
2. A unified protocol for the setting of simulation parameters

ARTS properties can be set using Python assignment syntax, i.e.

::

    sensor.f_grid = np.array([1.0, 2.0, 3.0])

or they may be obtained from the :code:`data_provider` using the
appropriate get method when a concrete simulation is run. ARTS
properties impelement the required functionality to provide this
functionality to the user in an opaque manner.

Developer perspective
~~~~~~~~~~~~~~~~~~~~~

For the developer, the purpose of ARTS properties is to provide
meta-programming functionality to simplify the handling of ARTS-related
variables. It provides default methods for the setting of ARTS properties
as well as default :code:`setup` and :code:`get_data` methods.

Additionally, the :code:`arts_object` provides functionality for
the propagation of dimension information throughout :code:`parts`. This
is important to enable testing consistency of user-provided data.

Reference
=========
"""
import numpy as np
import inspect
from abc import ABCMeta, abstractmethod, abstractproperty
from typhon.arts.workspace.variables import WorkspaceVariable

################################################################################
# The Dimension class
################################################################################

class Dimension:
    """
    Service class to keep track of related dimensions throughout an ARTS
    simulation.

    A consistent ARTS simulation setup requires provided data to have the
    same sizes along related dimensions. The :code:`Dimensions` class
    provides a symbolic representation of these dimensions and provides
    functionality to propagate information on the values of these dimensions
    as data is provided by the user.
    """

    #
    # Singleton classes describing dimensions
    #
    class P:
        """Singleton class representing the pressure dimension."""
        pass

        @classmethod
        def __repr__(self):
            return "pressure"

    class Lat:
        """Singleton class representing the latitude dimension."""
        pass

        @classmethod
        def __repr__(self):
            return "latitude"

    class Lon:
        """Singleton class representing the longitude dimension."""
        pass

        @classmethod
        def __repr__(self):
            return "longitude"

    class Atm:
        """
        Singleton class representing the number of dimensions of the
        Atmosphere.
        """
        pass

        @classmethod
        def __repr__(self):
            return "atmosphere"

    class Los:
        """ Singleton class representing the line-of-sight dimension. """
        pass

        @classmethod
        def __repr__(self):
            return "line-of-sight"

    class Frq:
        """Singleton class representing the number of frequencies."""
        pass

        @classmethod
        def __repr__(self):
            return "frequency"

    class Joker:
        """
        Singleton class representing dimensions that can take an
        arbitrary value.
        """
        pass

    dimensions = [P, Lat, Lon, Atm, Los, Frq, Joker]

    def __init__(self):
        """
        Create a dimensions object.
        """
        self.dimensions = {}

    def infer(self, dim):
        """
        Infer the value of a dimension.

        Parameters:

            dim: Singleton class representing the dimension to lookup.

        Returns:

            None if no value has been deduced for the requested dimension.

            Otherwise a tuple :code:`(value, who)` consisting of an integer
            value :code:`value` representing the deduced value of the requested
            dimension as well as a :code:`string` :code:`who` indicating
            the variable from which the dimensions has been deduced
        """
        if dim in self.dimensions:
            return self.dimensions[dim]
        else:
            return None

    def deduce(self, dim, value, who):
        """
        Add a deduction of a value of a given dimension.

        This function should be called when data has been provided by the
        user that allows the deduction of the size of a given dimension
        if no previous such value has been deduced, i.e. the :code:`infer`
        method for the same dimension returns :code:`None`

        Parameters:

            dim: Singleton class representing the dimension to lookup.

            value(int): The value of the dimension that could be deduced.

            who(str): Name of the variable from which the value has been
                deduced.

        """
        if dim in self.dimensions:
            other, _ = self.dimensions[dim]
            if not other == value:
                raise Exception("A conflicting value for dimension {0} has "\
                                "aready been deduced.".format(dim))
        else:
            self.dimensions[dim] = (value, who)

    def link(self, other):
        """
        Link to :code:`Dimension` objects.

        This registers the :code:`self` :code:`Dimension` object as master of the
        :code:`other` :code:`Dimensions` object. Deduced dimensions values from
        the *master* dimension are propagated to the *slave* dimensions but not
        vice versa. This allows having dimensions with different values in the
        *slave* objects while still ensuring consistency with the *master* object.

        Parameters:

            other(:code:`Dimensions`): The *slave* dimension to link to this
            dimension.

        """
        # Check consistency of dimension objects.
        for k in self.dimensions:
            if k in other.dimensions:
                v1, who1 = self.dimensions[k]
                v2, who2 = other.dimensions[k]

                if v1 == v2:
                    continue

                if v1 == 1 or v2 == 1:
                    continue

                s = r"Contradictory dimensions deduced for dimension {0}:" \
                    "{1} deduced from {2} and {3} deduced from {4}."\
                    .format(k, v1, who1, v2, who2)
                raise Exception(s)
            else:
                other.dimensions[k] = self.dimensions[k]

        self.dimensions = other.dimensions

################################################################################
# ARTS properties
################################################################################

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
        return obj.shape
    elif type(obj) == list:
        return (len(obj),) + get_shape(obj[0])
    else:
        return ()

def broadcast(shape, obj):
    """
    Broadcast an array or a list to a provided shape.

    Parameters:

        shape(tuple): Tuple of :code:`int` describing the shape to broadcast
        the provided object to.

        obj(object): The object to broadcast to the provided shape. Either a
        numpy array or a (nested) list.
    """
    if type(obj) == np.ndarray:
        return np.broadcast_to(obj, shape)
    elif type(obj) == list:
        if len(obj) == 1:
            return shape[0] * [broadcast(shape[1:], obj[0])]
        else:
            return [broadcast(shape[1:], nested) for nested in obj]
    else:
        if shape == ():
            return obj
        else:
            raise Exception("Degree of nesting of list is inconsitent with"\
                            " length of the provided shape tuple.")


def arts_property(group, shape = (), wsv = None):
    """
    The :code:`arts_property` decorator.

    This decorator turns a function defintion into an ARTS property. Its useage
    is similar to the :code:`@property` decorator in Python. The ARTS property
    decorator provides default getter, setter, :code:`setup` and :code:`get_data`
    methods. The setter as well as the :code:`setup` and :code:`get_data` methods
    can be overwritten to allow specializing the handling of specific variables.
    For more details see the :class:`ArtsProperty` class.
    ::
        @arts_property
        def f_grid:
            return np.array([187e9]) # The default value

        @f_grid.setter
        def set_f_grid(self, value):
            print("Setting the f_grid.")
            self.value = value

        @f_grid.setter
        def set_f_grid(self, value):
            print("Setting the f_grid.")
            self.value = value

        @f_grid.setup
        def setup_f_grid(self, value):
            print("Customg setup method.")

        @f_grid.get_data
        def get_data_f_grid(self, value):
            print("Customg setup method.")

    Parameters:

        group(str): The ARTS group to which the variable should belong.

        shape(tuple): Tuple describing the expected shape of the variable.

        wsv(typhon.arts.workspace.WorkspaceVariable): The workspace variable
        corresponding to the ARTS property.
    """
    class ArtsPropertySpecialization(ArtsProperty):
        def __init__(self, fdefault):
            super().__init__(fdefault, group, shape, wsv)
    return ArtsPropertySpecialization

class ArtsProperty:
    """
    The :code:`ArtsProperty` class that implements the main functionality of
    ARTS properties.

    An ARTS property is an abstract representation of a parameter of an ARTS
    simulation. This parameter can be set to a fixed value by the user during
    the setting up of the simulation. In this case the property is said to
    be *fixed*. Otherwise, and if no default value for the parameter is set,
    the value will be requested from the data provider.

    The :code: `ArtsProperty` class implements the follwing functionality:

        1. Provide a default setter, that checks consistency with expected ARTS
        group and shape.

        2. Provide a default setup method that sets the value of the
        corresponding ARTS WSV if the value of the property is set to fixed.


    The :code:`ArtsProperty` class is to be used through the :code:
    `arts_property` decorator, which returns a subclass of this class that
    fixes the :code:`group, shape` and :code:`wsv` values of the :code:`__init__`
    method.
    """
    def __init__(self, fdefault, group, shape, wsv):
        """
        Create a :code:`ArtsProperty` instance.

        Parameters:

            fdefault: Default value for the ARTS Property or :code:`None` if no
            reasonable default value can be provided.

            group(str): Name of the ARTS group the value belongs to.

            shape(tuple): Tuple describing the expected shape of the variable.
            Set to :code:`None` if no reasonable expected shape can be
            specified.

            wsv(typhon.arts.workspace.WorkspaceVariable): Workspace variable
            corresponding to this ARTS property or :code:`None` if no such
            WSV exists.
        """
        self.group = group
        self.shape = shape
        self.wsv = wsv

        self.fixed = False
        self.value = None

        self.fsetup    = None
        self.fget_data = None
        self.fset      = None

        self.name  = fdefault.__name__
        self.value = fdefault()

    def get_name(self, owner, separator = "_"):
        """
        Return a qualified name of the ARTS property.

        This return a name of the ARTS property prefixed by the name of the
        owner is this object possesses a :code:`name` attribute.

        Parameters:

            owner: The object instance that this ARTS property belongs
            to.

            separator(str): Char to use to separater the owners name and
            the name of the ARTS property.

        """
        if hasattr(owner, "name"):
            name = getattr(owner, "name") + separator
        else:
            name = ""
        name += self.name
        return name

    def setup(self, fsetup):
        """
        Decorator to set the setup method of the ARTS property. The expected
        call signature of the setup method is :code:`setup(self, obj, ws)`
        where :code:`obj` is the owner of the ARTS property. Providing of the
        :code:`obj` argument of the :code:`setup` method is currently necessary
        to allow the checking of dimensions.

        Parameters:

            fsetup(function): The customized setup function to use for this
            ARTS property.

        """
        self.fsetup = setup

    def get_data(self, fget_data):
        """
        Decorator to set the :code:`get_data` method of the ARTS property.
        The expected call signature of the :code:`get_data` method is
        :code:`get(self, obj, ws, *args, **args)` where :code:`obj` is
        the owner of the ARTS property. Providing of the :code:`obj` argument
        of the :code:`setup` method is currently necessary to allow the checking
        of dimensions.

        Parameters:

            fsetup(function): The customized :code:`get_data` method to use for
            this ARTS property.

        """
        self.fget_data = fget_data

    def setter(self, fset):
        """
        Decorator to set the setter to set the value of this ARTS property.
        The expected call signature is :code:`set(self, value)` and should
        set the :code:`value` attribute of the :code:`ArtsProperty` object
        :code:`self`.

        Parameters:

            fset(function): The customized setter to use for this
            :code:`ArtsPorperty`

        """
        self.fset = fset

    def check_and_broadcast(self, value, owner):
        """
        check the shape of a given variable against a symbolic shape specification.

        Parameters:

            value(array or list): the object of which to check the shape.

            who(str): name of the variable of which the shape is checked.

        Raises:

            Exception if the shape of :code:`value` is inconsitent with the shape
            specification the ARTS property.

        """
        who = self.get_name(owner, separator = ".")
        shape = get_shape(value)


        # catch inconsistent number of dimensions.
        if not len(shape) == len(self.shape):
            s = "The provided value for {0} has {1} dimensions but {2} were"\
                " expected.".format(who, len(shape), len(self.shape))
            raise Exception(s)

        # deduce and compare dimensions.
        error = "The provided value has dimension {1} along axis {0} but {2}" \
                "were expected."

        deduced = tuple()
        for i in range(len(self.shape)):

            # Fixed dimension
            if type(self.shape[i]) == int:
                deduced += (shape[i],)
                if shape[i] == self.shape[i]:
                    continue
                raise Exception(error.format(i, shape[i], self.shape[i]))

            # Symbolic dimension
            elif self.shape[i] in Dimension.dimensions:
                d = owner.dimensions.infer(self.shape[i])
                if d is None:
                    owner.dimensions.deduce(self.shape[i], shape[i], who)
                    deduced += (shape[i],)
                else:
                    n, who2 = d
                    deduced += (n,)
                    if n == shape[i] or shape[i] == 1:
                        continue
                    s = "The value provided for the {0} property was expected"\
                        " to match the {1} dimension of the simulation  along"\
                        " axis {2} but this is not the case. The value of the"\
                        " {1} dimension has been deduced to be to be {3}"\
                        " from the value of the {4} property."
                    raise Exception(s.format(who, self.shape[i], i, n, who2))
            else:
                raise Exception("Shape specification should consist of either"\
                                " integers or symbolic dimensions.")

        value = broadcast(deduced, value)
        return value

    def __get__(self, obj, objtype = None):
        if obj is None:
            return self

        return self.value

    def __set__(self, owner, value):
        if not self.fset is None:
            self.fset(self, value)
            return None

        converted = WorkspaceVariable.convert(self.group, value)

        if converted is None:
            raise Exception("Provided value of type {0} cannot be converted"
                            " to ARTS group {1}".format(type(value),
                                                        self.group))
        value = converted

        if not self.shape is None:
            value = self.check_and_broadcast(value, owner)

        self.fixed = True
        self.value = value

    def _setup(self, *args, **kwargs):
        """
        Run the default or customized :code:`setup` method of the
        :code:`ArtsProperty` object.
        """
        if self.fsetup is None:
            self._default_setup(*args)
        else:
            self.fsetup(*args, **kwargs)

    def _get_data(self, owner, ws, data_provider, *args, **kwargs):
        """
        Run the default or customized :code:`get_data` method of the
        :code:`ArtsProperty` object.
        """
        if self.fget_data is None:
            self._default_get_data(owner, ws, data_provider,
                                   *args, **kwargs)
        else:
            self.fget_data(owner, ws, data_provider,
                           *args, **kwargs)

    def _default_setup(self, owner, ws):
        """
        The default :code:`setup` method of the :code:`ArtsProperty`.
        This method simply checked whether the value of the :code:`ArtsProperty`
        was set to a fixed value. If this is the case and the :code:`wsv`
        attribute contains an associated WSV, the WSVs value in the
        given workspace :code:`ws` is set.

        Parameters:

            owner(obj): The object that this :code:`ArtsProperty` belongs
            to.

            ws(arts.typhon.workspace.Workspace): Ths workspace which to
            setup.
        """
        if self.wsv and self.fixed:
            if self.wsv.name in owner._wsvs:
                owner._wsvs[wsv.name].value = owner.value
            else:
                setattr(ws, self.wsv.name, self.value)

    def _default_get_data(self, owner, ws, data_provider, *args, **kwargs):
        """
        The default :code:`get_data` method. It performs the following steps:

            1. Check if the :code:`data_provider` provides a get method for the
            :code:`ArtsProperty`. If so, then set the associated WSV to this
            value (if given).

            2. Check if the value of the :code:`ArtsProperty` is fixed. In this
            case it has been set already during setup, so no additional action
            is required.

            3. Check if a default value has been provided for the
            :code:`ArtsProperty`. In this case use this value.

            4. Throw an exception if all of the previous steps were unsuccessful
            in determining the value of the :code:`ArtsProperty`.
        """
        if self.wsv and not self.fixed:
            getter_name = "get_" + self.name

            # Try to get value from provider.
            if hasattr(data_provider, getter_name):
                f = getattr(data_provider, "get_" + self.name)
                value = f(*args, **kwargs)

                converted = WorkspaceVariable.convert(self.group, value)

                if converted is None:
                    raise Exception("Provided value of type {0} cannot be"\
                                    " converted to ARTS group {1}"\
                                    .format(type(value), self.group))

                if not self.shape is None:
                    value = self.check_and_broadcast(value, owner)
                    owner.set_wsv(ws, self.wsv, value)

            # Check if there's a default value.
            elif not self.default is None:
                owner.set_wsv(ws, self.wsv, value)

            # No value - throw exception
            else:
                raise Exception("Neither a default value nor a get method "
                                " has been provided for the ARTS property"
                                "{0}.".format(self.get_name(owner, ".")))


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
        if not isinstance(x, prop.expected_type):
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

def add_property(obj, name, dims, t):
    """
    Add an ARTS property to an existing object.

    Parameters:

        obj(:code:`object`): The object to add the property to.

        name(:code: `str`): Name of the property to create

        dims(:code:`tuple`): Tuple describing the expected dimensions
        of the property.

        t(:code:`type`): The expected type of the property.
    """
    getter = make_getter(name)
    setter = make_setter(name)
    prop = property(getter, setter, name)
    setattr(type(obj), name, prop)

    ph = PlaceHolder(name, dims, t)
    setattr(obj, "_" + name, ph)

def is_arts_property(obj):
    return isinstance(obj, ArtsProperty)

class ArtsObjectReplacement:
    """
    The :ArtsObject: class provides a base class for objects that bundle
    ARTS workspace variables and functionality into a conceptual unit. The
    class provides service functions that automate the handling of ARTS
    properties.
    """

    def __init__(self):
        """
        Create an :code:`ArtsObject` instance.
        """
        self._wsvs      = {}
        self.dimensions = Dimension()

    def setup_arts_properties(self, ws):
        """
        Run the :code:`setup` method for all ARTS properties of the class.

        Parameters:

            ws(typhon.arts.workspace): The workspace for which to setup
            the simulation.

        """
        for _, ap in inspect.getmembers(type(self), is_arts_property):
            ap._setup(self, ws)

    def get_data_arts_properties(self, ws, data_provider, *args, **kwargs):
        """
        Run the :code:`get_data` method for all ARTS properties of this
        object.

        Parameters:

            ws(typhon.arts.workspace): The workspace for which to setup
            the simulation.

            data_provider(obj): The :code:`data_provider` providing the
            data for the simulation.

            *args: Additional parameters to be passed on to the data
            provider.

            **kwargs: Additional keyword arguments to be passed to the data
            provider.
        """
        for _, ap in inspect.getmembers(type(self), is_arts_property):
            ap._get_data(self, ws, data_provider, *args, **kwargs)

    def set_wsv(self, ws, wsv, value):
        """
        Sets the given private WSV :code:`wsv` of the object or if the
        object doesn't have a private copy of :code:`wsv` then sets :code:`wsv`
        on the workspace :code:`ws`.

        Parameters:

            ws(typhon.arts.workspace.Workspace): The workspace in which to set
            the WSV.

            wsv(typhon.arts.workspace.variables.WorkspaceVariable): The
            variable to set.

            value(obj): The value to set the WSV :code:`wsv` to.
        """
        if wsv.name in self._wsvs:
            self._wsvs[wsv].ws    = ws
            self._wsvs[wsv].value = value
        else:
            setattr(ws, wsv.name, value)


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

            for nm, dim, t in ps:
                getter = make_getter(nm)

                # Check for custom setter.
                if nm + "_setter" in dct:
                    setter = dct[nm + "_setter"]
                else:
                    setter = make_setter(nm)

                prop = property(getter, setter, nm)
                dct[nm] = prop

        return super().__new__(cls, name, bases, dct)

