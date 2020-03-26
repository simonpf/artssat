""" Dimensions

The Dimensions class keeps track of the different dimensions in
an ARTS atmosphere. The purpose of this class is to allow the inferral
of dimensions from the provided data, but at the same time to be able
to identify inconsistencies.

Attributes:

    p(Dimension): Dimension object representing the size of the
        ARTS pressure grid
    lat(Dimension) Dimension object representing the size of the
        latitude grid
    lon(Dimension) Dimension object representing the size of the
        longitude grid
    joker(Joker) Dimension object representing a dimensions that
        can take an arbitrary value.

"""

class Dimension:
    """
    Dimensions of data arrays in ARTS.

    The required dimensions of most array data used in an ARTS
    simulation are fixed by the dimensions of the atmosphere.
    The Dimension class provides a symbolic representation of
    these dimensions in an ARTS simulation, so that these can
    be inferred and checked for consistency.

    Dimension objects are used to define the dimensions of
    ARTS data and keep track of the different values inferred
    from user-provided data.

    """
    def __init__(self, name):
        """
        Create dimension with given name.

        Parameters:
            name(str): The name of the dimension, e.g. "Pressure grid"

        """
        self.name = name
        self.deductions = {}

    def add_deduction(self, ws, value, who):
        """
        If a value of a dimension becomes apparent from data it
        should be signaled to the :code:`Dimension` object. The
        dimension object keeps track of these "deductions" for
        each workspace separately.

        Parameters:

            ws(pyarts.workspace.Workspace): Workspace object
                for which the dimension was deduced.

            value(Int): The deduced dimenions

            who(str): The name of the variable from which this
                value was deduced.

        """
        if not ws in self.deductions:
            self.deductions[ws] = [(value, who)]
        else:
            self.deductions[ws] += [(value, who)]

    def check(self, ws, n, who):
        """
        Check given value for consistency.

        Parameters:

            ws(pyarts.workspace.Workspace): The workspace for which
                to check the deduction.

            n(int): The value which to check against previous deduced values.

        Returns:

            True if this is the first deduction or is previous deuctions agree
            with the current value. False otherwise.

        """

        if not ws in self.deductions:
            self.deductions[ws] = (n, who)
            return True
        else:
            (m, who) = self.deductions[ws]
            return m == n

    def get_value(self, ws):
        """
        Get the deduced value of the dimension.

        Parameters:

            ws(pyarts.workspace.Workspace) The workspace object for
                which the value of the dimension should be deduced.

        Raises:

            Exception: If no or inconsistent values have been deduced.

        """

        self.check(ws)

        ds = self.deductions[ws]
        if len(ds) > 0:
            return ds[0][0]
        else:
            raise Exception("No value for dimension of the {0} has been "
                            "deduced.")

class Joker(Dimension):
    """
    A dimension that can take an arbitrary value.

    """
    def __init__(self, name):
        """
        Create dimension with given name.

        Parameters:
            name(str): The name of the dimension, e.g. "Pressure grid"

        """
        self.name = name
        self.deductions = {}

    def add_deduction(self, ws, value, who):
        """
        This function does nothing.

        """
        pass

    def check(self, *args, **kwargs):
        """
        This function does nothing.

        """
        pass

    def get_value(self, *args, **kwargs):
        """
        Raises an exception because it doesn't make sense to try
        an get a value from a joker dimension.

        Raises:

            Exception: If no or inconsistent values have been deduced.

        """

        raise Exception("Can't get the value of a joker dimension.")

p     = Dimension("pressure grid")
lat   = Dimension("latitude grid")
lon   = Dimension("longitude grid")
atm   = Dimension("atmospheric dimension")
los   = Dimension("line of sight dimension")
joker = Joker("This dimension can take any value.")
