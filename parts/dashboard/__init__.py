import threading
import tornado.ioloop

from bokeh.server.server                 import Server
from bokeh.application                   import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.plotting                      import figure, ColumnDataSource
from bokeh.models.widgets                import Div, Tabs, Panel
from bokeh.layouts                       import column, row

from parts.sensor import ActiveSensor, PassiveSensor


class BokehServer():
    """
    Handles the singleton Bokeh server used to serve visualisations
    if no jupyter notebook is open.
    """
    def __init__(self):
        self.server = None

    def __del__(self):
        if not self.server is None:
            self.server.io_loop.stop()
            self.server.stop()

    def start(self, application, port = 5000):

        if not self.server is None:
            self.server.io_loop.stop()
            self.server.stop()

        self.server = Server(application, port = port)

        try:
            self.server.io_loop.start()
        except:
            raise
        finally:
            print("Stopping server.")
            tornado.ioloop.IOLoop.instance().stop()
            self.server.stop()


standalone_server = BokehServer()

################################################################################
# Atmosphere
################################################################################


def plot_temperature(simulation, data_provider):

    fig = figure(title = "Temperature", width = 500)

    t = simulation.workspace.t_field.value.ravel()
    z = simulation.workspace.z_field.value.ravel() / 1e3

    fig.line(x = t, y = z)
    fig.circle(x = t, y = z)

    fig.xaxis.axis_label = "Temperature [K]"
    fig.yaxis.axis_label = "Altitude [km]"

    return fig

def plot_absorption_species(simulation, data_provider):

    fig = figure(title = "Absorption species", width = 500)

    for a in simulation.atmosphere.absorbers:

        i = a._wsv_index
        x = simulation.workspace.vmr_field.value[i, :, :, :].ravel()
        z = simulation.workspace.z_field.value.ravel() / 1e3

        fig.line(x = x, y = z)
        fig.circle(x = x, y = z)

    fig.xaxis.axis_label = "VMR [mol/m^3]"
    fig.yaxis.axis_label = "Altitude [km]"

    return fig


def plot_scattering_species(simulation, data_provider):
    pass


def make_atmosphere_panel(simulation, data_provider):
    c = column(plot_temperature(simulation, data_provider),
               plot_absorption_species(simulation, data_provider))
    return Panel(child = c, title = "Atmosphere", width = 600)

################################################################################
# Sensor
################################################################################

def make_sensor_plot(sensor):

    fig = figure(title = sensor.name, width = 500)

    if isinstance(sensor, ActiveSensor):
        z = sensor.range_bins
        x = sensor.y
        z = 0.5 * (z[1:] + z[:-1])

        for i in range(sensor.y.shape[1]):
            x = sensor.y[:, i]
            y = z.ravel() // 1e3
            fig.line(x = x, y = y)
            fig.circle(x = x, y = y)

        fig.xaxis.axis_label = "Radar reflectivity [{0}]"\
                            .format(sensor.iy_unit)
        fig.yaxis.axis_label = "Altitude [km]"

    if isinstance(sensor, PassiveSensor):

        x = sensor.f_grid
        y = sensor.y

        for i in range(sensor.y.shape[1]):
            fig.line(x = x, y = y[:, i])
            fig.circle(x = x, y = y[:, i])


    return fig


def make_sensor_panel(sensors):
    c = column(*[make_sensor_plot(s) for s in sensors],
               sizing_mode = "fixed")
    return Panel(child = c, title = "Measurements", width = 600)

def make_document_factory(simulation):

    def make_document(doc):

        doc.title = "parts dashboard"

        sensor_p     = make_sensor_panel(simulation.sensors)
        atmosphere_p = make_atmosphere_panel(simulation, simulation.data_provider)

        t = Tabs(tabs = [sensor_p,
                         atmosphere_p])
        doc.add_root(t)

    return make_document

def dashboard(simulation):
    apps = {'/': Application(FunctionHandler(make_document_factory(simulation)))}
    standalone_server.start(apps, port = 8880)

