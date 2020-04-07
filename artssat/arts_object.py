"""
artssat.arts_object
-------------------

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
the propagation of dimension information throughout :code:`artssat`. This
is important to enable testing consistency of user-provided data.

Reference
=========
"""
import numpy as np
import inspect
from abc import ABCMeta, abstractmethod, abstractproperty
from pyarts.workspace.variables import WorkspaceVariable, group_names
from pyarts.workspace.variables import workspace_variables as wsv
from pyarts.workspace import Workspace

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
        @classmethod
        def __repr__(self):
            return "pressure"

    class Lat:
        """Singleton class representing the latitude dimension."""
        @classmethod
        def __repr__(self):
            return "latitude"

    class Lon:
        """Singleton class representing the longitude dimension."""
        @classmethod
        def __repr__(self):
            return "longitude"

    class Atm:
        """
        Singleton class representing the number of dimensions of the
        Atmosphere.
        """
        @classmethod
        def __repr__(self):
            return "atmosphere"

    class Los:
        """ Singleton class representing the line-of-sight dimension. """
        @classmethod
        def __repr__(self):
            return "line-of-sight"

    class Obs:
        """
        Singleton dimension representing the number of measurement
        blocks.
        """
        @classmethod
        def __repr__(cls):
            return "measurement block"

    class Frq:
        """Singleton class representing the number of frequencies."""
        @classmethod
        def __repr__(self):
            return "frequency"

    class Joker:
        """
        Singleton class representing dimensions that can take an
        arbitrary value.
        """
        pass

    dimensions = [P, Lat, Lon, Atm, Los, Obs, Frq, Joker]

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
        if dim is Dimension.Joker:
            return None

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
        if dim is Dimension.Joker:
            return None

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
    if hasattr(obj, "shape"):
        return obj.shape
    elif type(obj) == list:
        if obj == []:
            return (0,)
        else:
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

    if shape == get_shape(obj):
        return obj

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


def arts_property(group, shape = None, wsv = None, optional = False):
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

        wsv(pyarts.workspace.WorkspaceVariable): The workspace variable
        corresponding to the ARTS property.
    """
    class ArtsPropertySpecialization(ArtsProperty):
        def __init__(self, fdefault):
            super().__init__(fdefault, group, shape, wsv, optional)
    return ArtsPropertySpecialization

ws = Workspace(verbosity = 0)

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
    def __init__(self, fdefault, group, shape, wsv, optional):
        """
        Create a :code:`ArtsProperty` instance.

        Parameters:

            fdefault: Default value for the ARTS Property or :code:`None` if no
            reasonable default value can be provided.

            group(str): Name of the ARTS group the value belongs to.

            shape(tuple): Tuple describing the expected shape of the variable.
            Set to :code:`None` if no reasonable expected shape can be
            specified.

            wsv(pyarts.workspace.WorkspaceVariable): Workspace variable
            corresponding to this ARTS property or :code:`None` if no such
            WSV exists.

            optional(Boolean): If True no Exception will be thrown if the
            data_provider doesn't provide a get method for this property.
        """
        self.group = group
        self.shape = shape
        self.optional = optional

        if not wsv is None:
            if type(wsv) == str:
                if hasattr(ws, wsv):
                    wsv = getattr(ws, wsv)
                else:
                    raise Exception("Workspace variable {0} associated with"\
                                    " ARTS property {1} does not exist."\
                                    .format(wsv, fdefault.__name__))
        self.wsv = wsv
        self.fsetup    = None
        self.fget_data = None
        self.fset      = None

        self.name  = fdefault.__name__
        self.fdefault = fdefault

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

    def check_and_convert(self, value):
        """
        Checks type of :code:`value` against the group specification of this
        ARTS property contained in :code:`self.group`.

        The group specification can be a single string in which case this
        function will try to convert the provided value to the given group
        using the :code:`convert` class method of
        :class:`pyarts.workspace.WorkspaceVariable`.

        It is also possible to specify a list of groups for the expected
        value. In this case this function simply checks whether the group
        inferred by :code:`WorkspaceVariable.get_group_id`
        group of the :code:``

        """
        if type(self.group) == str:
            converted = WorkspaceVariable.convert(self.group, value)

            if converted is None:
                raise Exception("Provided value of type {0} cannot be converted"
                                " to ARTS group {1}".format(type(value),
                                                            self.group))
            value = converted
        elif type(self.group) == list:
            g_i = WorkspaceVariable.get_group_id(value)
            g = group_names[g_i]
            if not g in self.group:
                raise Exception("Provided value of type {0} is not of any of "
                                " the expected ARTS groups {1}."\
                                .format(type(value), self.group))

        return value

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

        if self.shape is None:
            return value

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

    def __get__(self, owner, objtype = None):

        if owner is None:
            return self

        val = owner.__dict__["_" + self.name].value

        if val is None:
            val = self.fdefault(owner)

        return val

    def __set__(self, owner, value):
        if not self.fset is None:
            self.fset(owner, value)
            return None

        value = self.check_and_convert(value)

        if not self.shape is None:
            value = self.check_and_broadcast(value, owner)

        ph = owner.__dict__["_" + self.name]
        ph.fixed = True
        ph.value = value

        if not self.wsv is None:
            if not ph.workspace is None:
                owner.set_wsv(ph.workspace, self.wsv, value)

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
        ph = owner.__dict__["_" + self.name]
        ph.workspace = ws
        if self.wsv and ph.fixed:
            owner.set_wsv(ws, self.wsv, ph.value)

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
        ph = owner.__dict__["_" + self.name]
        if not ph.fixed:
            getter_name = "get_" + self.get_name(owner, separator = "_")

            default = self.fdefault(owner)

            # Try to get value from provider.
            if hasattr(data_provider, getter_name):
                f = getattr(data_provider, getter_name)
                value = f(*args, **kwargs)

                value = self.check_and_convert(value)

                if not self.shape is None:
                    value = self.check_and_broadcast(value, owner)

                if self.wsv:
                    owner.set_wsv(ws, self.wsv, value)

                ph = owner.__dict__["_" + self.name]
                ph.value = value

            # Check if there's a default value.
            elif not default is None:
                ph = owner.__dict__["_" + self.name]
                ph.value = default

                if self.wsv:
                    owner.set_wsv(ws, self.wsv, default)

            # No value - throw exception
            elif not self.optional:
                raise Exception("Neither a default value nor a get method "
                                " has been provided for the ARTS property "
                                "{0}.".format(self.get_name(owner, ".")))
            else:
                return


class PlaceHolder:
    """
    Data that is required for an ARTS simulation and can be set
    either to a fixed value, or taken from a data provider.
    """
    def __init__(self):
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
        self.workspace = None

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

class ArtsObject:
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

        for _, ap in inspect.getmembers(type(self), is_arts_property):
            self.__dict__["_" + ap.name] = PlaceHolder()

    def setup_arts_properties(self, ws):
        """
        Run the :code:`setup` method for all ARTS properties of the class.

        Parameters:

            ws(pyarts.workspace): The workspace for which to setup
            the simulation.

        """
        for _, ap in inspect.getmembers(type(self), is_arts_property):
            ap._setup(self, ws)

    def setup(self, ws):
        self.setup_arts_properties(ws)

    def get_data_arts_properties(self, ws, data_provider, *args, **kwargs):
        """
        Run the :code:`get_data` method for all ARTS properties of this
        object.

        Parameters:

            ws(pyarts.workspace): The workspace for which to setup
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

    def get_data(self, ws, data_provider, *args, **kwargs):
        self.get_data_arts_properties(ws, data_provider, *args, **kwargs)

    def set_wsv(self, ws, wsv, value):
        """
        Sets the given private WSV :code:`wsv` of the object or if the
        object doesn't have a private copy of :code:`wsv` then sets :code:`wsv`
        on the workspace :code:`ws`.

        Parameters:

            ws(pyarts.workspace.Workspace): The workspace in which to set
            the WSV.

            wsv(pyarts.workspace.variables.WorkspaceVariable): The
            variable to set.

            value(obj): The value to set the WSV :code:`wsv` to.
        """
        if wsv.name in self._wsvs:
            self._wsvs[wsv.name].ws    = ws
            self._wsvs[wsv.name].value = value
        else:
            setattr(ws, wsv.name, value)

    def update_wsv(self, wsv, value):
        """
        Updates the value of a given WSV considering private WSVs of the owner.
        This requires that the variable has already been set to a value during
        so that is contains a reference to the workspace in which it is used.

        Parameters:

            wsv(pyarts.workspace.variables.WorkspaceVariable): The
            variable to set.

            value(obj): The value to set the WSV :code:`wsv` to.
        """
        if wsv.name in self._wsvs:
            self._wsvs[wsv.name].value = value
        else:
            wsv.value = value

    def call_wsm(self, ws, wsm):
        """
        Call workspace method on the given workspace.

        This method replaces inputs of the workspace variable with the private
        WSMs of the object. After execution of the method the results are
        copied to the private WSMs of the object.

        Parameters:

            ws(pyarts.workspace.Workspace): The workspace on which to
            execute the method.

            wsm(pyarts.workspace.WorkspaceMethod): The workspace method
            to execute.

        """
        args = self.get_wsm_kwargs(wsm)
        wsm.call(ws, **args)

        # Copy output
        for i in wsm.outs:
            name = WorkspaceVariable.get_variable_name(i)
            if name in self._wsvs and not i in wsm.ins:
                ws.Copy(self._wsvs[name], wsv[name])

    def _create_private_wsvs(self, ws, names):
        """
        Create private copies of given WSV names.

        Parameters:

            ws(:code:`pyarts.workspace.Workspace`): A workspace instance
            on which to create the workspace variables.

            names(list): List of strings containing the names of the workspace
            variables to create.
        """
        for name in names:
            wsv = ws.__getattr__(name)
            wsv_private = ws.create_variable(wsv.group,
                                             self.name + "_" + name)
            self._wsvs[name] = wsv_private

    def get_wsm_kwargs(self, wsm):
        """
        Generate a list of arguments to the given ARTS workspace method
        :code:`wsm` for which the sensor related input parameters are
        replace by the ones of this sensor. This is done by checking
        whether the input argument name is in the sensors :code:`_wsv`
        dictionary and if so replacing the argument.

        Parameters:

           wsm(pyarts.workspace.methods.WorkspaceMethod): The ARTS
               workspace method object for which to generate the input
               argument list.

        Returns:

            The list of input arguments with sensor specific input arguments
            replaced by the corresponding WSVs of the sensor.

        """
        kwargs = {}
        for i in wsm.ins:
            name = WorkspaceVariable.get_variable_name(i)
            if name in self._wsvs:
                kwargs[name] = self._wsvs[name]
            else:
                kwargs[name] = wsv[name]
        return kwargs



