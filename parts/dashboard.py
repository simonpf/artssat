import threading

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

        def start_server():
            try:
                self.server.io_loop.start()
            except Exception as e:
                print(str(e))
                print("shutting down server.")
                
            print("Server done.")

        thread = threading.Thread(target = start_server)
        thread.deamon = True
        thread.start()

standalone_server = BokehServer()

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
        fig.output_backend = "svg"

    if isinstance(sensor, PassiveSensor):

        x = sensor.f_grid
        y = sensor.y

        for i in range(sensor.y.shape[1]):
            fig.line(x = x, y = y[:, i])
            fig.circle(x = x, y = y[:, i])


    return fig


def make_sensor_tab(sensors):

    c = row(*[make_sensor_plot(s) for s in sensors],
            sizing_mode = "fixed")
    return Tabs(tabs = [Panel(child = c, title = "Measurements", width = 600)])

def make_document_factory(simulation):

    def make_document(doc):
        doc.title = "parts dashboard"
        t = make_sensor_tab(simulation.sensors)
        doc.add_root(t)

    return make_document

def dashboard(simulation):
    apps = {'/': Application(FunctionHandler(make_document_factory(simulation)))}
    standalone_server.start(apps, port = 8880)



