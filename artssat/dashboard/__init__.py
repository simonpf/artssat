import numpy as np

import threading

from bokeh                               import events
from bokeh.colors                        import RGB
from bokeh.server.server                 import Server
from bokeh.application                   import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.plotting                      import figure, ColumnDataSource, curdoc
from bokeh.models.widgets                import Div, Tabs, Panel, Select, CheckboxGroup, \
    CheckboxButtonGroup, RadioButtonGroup
from bokeh.models.glyphs                 import Image
from bokeh.models                        import Range1d, LinearColorMapper, ColorBar, \
    BasicTicker, PrintfTickFormatter, FuncTickFormatter, Rect
from bokeh.models.callbacks              import CustomJS
from bokeh.layouts                       import column, row
from bokeh.palettes                      import RdBu, Inferno

from artssat.sensor import ActiveSensor, PassiveSensor


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
        self.server.io_loop.start()


standalone_server = BokehServer()

################################################################################
# Profile plot
################################################################################

class ProfilePlot:
    """
    A profile plot displays the height distribution of an atmospheric quantity
    with respect to either altitude or pressure.
    """

    def __init__(self, z, p, title, name, width = 500):

        def update_y_variable(attrname, old, new):

            if self.y_axis_quantity.active == 0:
                self.y = self.z
                self.y_range = Range1d(start = self.z[0], end = self.z[-1])

            if self.y_axis_quantity.active == 1:
                self.y = self.p
                self.y_range = Range1d(start = self.p[-1], end = self.p[0])

            for s in self.sources:
                s.data["y"] = self.y

            fig = curdoc().get_model_by_name(self.name).children[0]
            fig.y_range = self.y_range

        def update_x_scale(attrname, old, new):
            if self.x_axis_scale.active == 1:
                for s in self.sources:
                    s.data["x"] = np.log10(s.data["x_"])
            else:
                for s in self.sources:
                    s.data["x"] = s.data["x_"]

        def update_y_scale(attrname, old, new):
            if self.y_axis_scale.active == 1:
                for s in self.sources:
                    s.data["y"] = np.log10(self.y)
            else:
                for s in self.sources:
                    s.data["y"] = self.y

        self.z = np.copy(z)
        self.p = np.copy(p)
        self.y = z
        self.quantities = []

        self.name = name

        self.x_axis_scale = RadioButtonGroup(labels = ["linear", "log"],
                                             active = 0)
        self.x_axis_scale.on_change("active", update_x_scale)

        self.y_axis_scale = RadioButtonGroup(labels = ["linear", "log"],
                                             active = 0)
        self.y_axis_scale.on_change("active", update_y_scale)

        self.y_axis_quantity = RadioButtonGroup(labels = ["altitude", "pressure"],
                                                active = 0)
        self.y_axis_quantity.on_change("active", update_y_variable)

        self.sources = []

        self.fig = figure(title = title, width = width, height = 500)

    def add_quantity(self, x, line_kwargs = {}):
        self.sources += [ColumnDataSource(data = dict(x = x,
                                                      x_ = x,
                                                      y = self.y))]
        self.fig.line("x", "y", source = self.sources[-1], **line_kwargs)

    @property
    def doc(self):
        d1 = Div(text = "y-axis variable:", width = 200)
        d2 = Div(text = "x-axis scale:", width = 200)
        d3 = Div(text = "y-axis scale:", width = 200)
        return column(self.fig,
                      row(d1, self.y_axis_quantity),
                      row(d2, self.x_axis_scale),
                      row(d3, self.y_axis_scale),
                      name = self.name)

class RetrievalResultPlot:
    def __init__(self, z, p, retrieval):
        self.z = z
        self.p = p

        y = z

        self.retrieval = retrieval
        self.retrieval_quantities = retrieval.retrieval_quantities

        self.figures = dict([(rq, ProfilePlot(self.z,
                                              self.p,
                                              rq.name,
                                              "rq_plot_" + rq.name,
                                              width = 400)) for
                             rq in self.retrieval_quantities])
        self.status = dict([(rq, True) for rq in self.retrieval_quantities])
        self.sources_x  = dict([(rq, []) for rq in self.retrieval_quantities])
        self.sources_xa = dict([(rq, []) for rq in self.retrieval_quantities])
        self.sources_x0 = dict([(rq, []) for rq in self.retrieval_quantities])
        self.lines_x  = dict([(rq, []) for rq in self.retrieval_quantities])
        self.lines_xa = dict([(rq, []) for rq in self.retrieval_quantities])
        self.lines_x0 = dict([(rq, []) for rq in self.retrieval_quantities])

        #
        # Get retrieval results as list.
        #
        if type(self.retrieval.results) is list:
            results = self.retrieval.results
        else:
            results = [self.retrieval.results]

        self.colors = Inferno[len(results) + 2][1:-1]

        #
        # Add plots for each retrieval quantity.
        #

        for i, rr in enumerate(self.retrieval.results):
            for rq in self.retrieval_quantities:
                xa = rr.get_result(rq, attribute = "xa")
                x  = rr.get_result(rq, attribute = "x")
                x0 = rr.get_result(rq, attribute = "x0")

                if xa is None:
                    continue

                self.sources_x[rq]  += [ColumnDataSource(data =
                                                         dict(x = x, y = y))]
                self.sources_xa[rq] += [ColumnDataSource(data =
                                                         dict(x = xa, y = y))]
                self.sources_x0[rq] += [ColumnDataSource(data =
                                                         dict(x = x0, y = y))]

                fig = self.figures[rq]
                fig.add_quantity(x, line_kwargs = dict(line_color = self.colors[i],
                                                       line_width = 2))
                if i == 0:
                    fig.add_quantity(xa, line_kwargs = dict(line_dash = "dashed",
                                                            line_color = self.colors[i],
                                                            line_width = 2))
                #fig.add_quantity(x0, line_kwargs = dict(line_dash  = "dashdot",
                #                                        line_color = self.colors[i],
                #                                        line_width = 2))


        self.plots = row(*[self.figures[k].doc for k in self.figures],
                         name = "retrieval_quantities")

        def update_plots(attrname, old, new):
            state = self.checks.active
            plots = curdoc().get_model_by_name('retrieval_quantities')

            print(state)
            for i, rq in enumerate(self.retrieval_quantities):
                if i in state:
                    if self.status[rq] == True:
                        continue
                    else:
                        self.status[rq] = True
                        plots.children.append(self.figures[rq].doc)
                else:
                    if self.status[rq] == False:
                        continue
                    else:
                        fig = curdoc().get_model_by_name("rq_plot_" + rq.name)
                        self.status[rq] = False
                        plots.children.remove(fig)

            print(plots.children)
            print(self.status)


        #
        # Checkbox button to hide plots
        #

        labels  = [rq.name for rq in self.retrieval.retrieval_quantities]
        active   = list(range(len(labels)))
        self.checks = CheckboxButtonGroup(labels = labels, active = active)
        self.checks.on_change("active", update_plots)


    def make_doc(self):
        title = Div(text = "<h2>Retrieved quantities</h2>")
        c = column(title, self.checks, self.plots)
        return c

class AVKPlot:
    def __init__(self, simulation):
        self.simulation = simulation
        self.sources = {}

        if not type(simulation.retrieval.results) is list:
            result = simulation.retrieval.results
        else:
            result = simulation.retrieval.results[-1]

        self.figures = {}
        self.images  = {}

        rqs = simulation.retrieval.retrieval_quantities
        self.panels = [self._make_avk_plot(result, rq) for rq in rqs]



    def _make_avk_plot(self, result, rq):
        avk = result.get_avk(rq)
        x   = np.arange(avk.shape[0])
        xx, yy  = np.meshgrid(x, x)
        self.sources[rq] = ColumnDataSource(data = dict(x = xx.ravel(),
                                                        y = yy.ravel(),
                                                        z = avk.ravel()))
        self.figures[rq] = figure(x_range = (x[0] - 0.5, x[-1] + 0.5),
                                  y_range = (x[0] - 0.5, x[-1] + 0.5),
                                  tools = ["box_zoom", "box_select", "reset"],
                                  width = 600,
                                  aspect_scale = 1.0,
                                  toolbar_location = "above")

        colors = RdBu[11]
        clim_high = max(avk.max(), -avk.min())
        clim_low  = min(-avk.max(), avk.min())

        mapper = LinearColorMapper(palette= colors,
                                   low = clim_low,
                                   high= clim_high)
        self.images[rq] = self.figures[rq].rect(source = self.sources[rq],
                                                x = "x",
                                                y = "y",
                                                fill_color = {'field' : 'z',
                                                              'transform' : mapper},
                                                line_color = None,
                                                width = 1,
                                                height = 1)
        #marker_source = ColumnDataSource(data = dict(x = np.array([x[0]]),
        #                                             y = np.array([x[0]])))

        #marker = Rect(x = "x",
        #              y = "y",
        #              fill_color = None,
        #              line_color = RGB(0.0, 0.0, 0.0),
        #              width = x[-1] - x[0],
        #              height = 1)
        #self.images[rq].add_glyph(marker_source, rect)

        #axis = self.figures[rq].xaxis
        #axis.major_label_overrides = dict(zip(axis.ticker,
        #                                      z[axis.ticker]))
        #axis = self.figures[rq].yaxis
        #axis.major_label_overrides = dict(zip(axis.ticker,
        #                                      z[axis.ticker]))

        def update(attr, old, new):
            print(old)
            print(new)
            print(self.sources[rq].selected.indices)

        #
        # The colorbar
        #

        color_bar = ColorBar(color_mapper = mapper,
                             #major_label_text_font_size ="5pt",
                             ticker = BasicTicker(desired_num_ticks = len(colors)),
                             formatter = PrintfTickFormatter(format="%.2f"),
                             label_standoff = 6,
                             border_line_color=None,
                             location=(0, 0))
        self.figures[rq].add_layout(color_bar, 'right')

        #
        # Single altitude
        #


        source = ColumnDataSource(data = dict(x = x, y = avk[0]))
        fig =  figure(tools = ["box_zoom", "box_select", "reset"],
                      width = 400,
                      y_range = [clim_low, clim_high],
                      toolbar_location = "above")
        line = fig.line(x = "x", y = "y", source = source)


        #
        # Plot callback
        #

        cb = CustomJS(args = dict(source_avk = self.sources[rq],
                                  #source_marker = marker_source,
                                  source = source), code =
                      """
                      var x = Number(cb_obj["x"]);
                      var y = Number(cb_obj["y"]);
                      console.log(x, y);
                      var i = Math.floor(y);
                      var j = i + 1;
                      var n = source.data["x"].length;
                      source.data["y"] = [];
                      for (var k = 0; k < n; k++) {
                          source.data["y"].push(source_avk.data["z"][i * n + k]);
                      }
                      source.change.emit();

                      console.log(source.data["x"].length);
                      console.log(source.data["y"].length);
                      """)
        self.figures[rq].js_on_event(events.Tap, cb)
        r = row(self.figures[rq], fig)

        return Panel(child = r, title = rq.name, width = 500)

    def make_doc(self):
        title = Div(text = "<h2>Averaging kernels</h2>")
        t = Tabs(tabs = self.panels)
        return column(title, t)

class ObservationPlot:

    def __init__(self, sensor):

        self.sensor = sensor
        self.observations = []

    def make_active_sensor_plot(self):

        fig = figure(title = self.sensor.name, width = 500)

        z = self.sensor.range_bins
        x = self.sensor.y
        z = np.ravel(0.5 * (z[1:] + z[:-1]))

        if self.observations == []:
            observations = [sensor.y]
        else:
            observations = self.observations

        for o in observations:
            x = o.reshape(z.size, -1)
            y = z // 1e3
            fig.line(x = x, y = o)
            fig.circle(x = x, y = o)

        fig.xaxis.axis_label = "Radar reflectivity [{0}]"\
                            .format(self.sensor.iy_unit)
        fig.yaxis.axis_label = "Altitude [km]"

        return fig

    def make_passive_sensor_plot(self):

        fig = figure(title = self.sensor.name, width = 500)

        x = self.sensor.f_grid
        y = self.sensor.y

        for o in self.observations:
            fig.line(x = x, y = o)
            fig.circle(x = x, y = o)

        return fig


    def add_observation(self, o):
        self.observations += [o]

    def make_doc(self):

        if isinstance(self.sensor, ActiveSensor):
            return self.make_active_sensor_plot()
        else:
            return self.make_passive_sensor_plot()


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
# Retrieval
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

def make_retrieval_panel(simulation):

    z = np.copy(simulation.workspace.z_field.value.ravel())
    p = np.copy(simulation.workspace.p_grid.value.ravel())

    retrieval_result_plot = RetrievalResultPlot(z, p, simulation.retrieval)
    r = retrieval_result_plot.make_doc()

    if type(simulation.retrieval.results) is list:
        results = simulation.retrieval.results
    else:
        results = [simulation.retrieval.results]

    if all([not r.avk is None for r in simulation.retrieval.results]):
        avks = AVKPlot(simulation)
        r.children += avks.make_doc().children

    return Panel(child = r, title = "Retrieval", width = 600)


    observation_plots = {}
    for s in simulation.sensors:
        observation_plots[s] = ObservationPlot(s)

    if type(simulation.retrieval.results) is list:

        plots = []
        for r in simulation.retrieval.results:

            for s in r.sensors:
                i, j = r.sensor_indices[s.name]
                observation_plots[s].add_observation(r.y[i : j])
                print(r.y[i:j])
                observation_plots[s].add_observation(r.yf[i : j])
                print(r.yf[i:j])

            plot = ProfilePlot(z, p)
            for q in r.retrieval_quantities:
                x = r.get_result(q)
                plot.add_quantity(x, r.name, dict(line_dash = "solid",
                                                  line_width = 3))
                xa = r.get_result(q, "xa")
                plot.add_quantity(xa, r.name, dict(line_dash = "dashed"))
                x0 = r.get_result(q, "x0")
                if not all(xa == x0):
                    plot.add_quantity(x0, r.name, dict(line_dash = "dotted"))

            plots += [plot]
    else:
        plot = ProfilePlot(z, p)
        for q in r.retrieval_quantities:
            x = r.get_result(q)
            plot.add(x, r.name, "blue")
        plots = [plot]

    title = Div(text = "<h2>Retrieved quantities</h2>")
    doms = [title] + [p.make_doc() for p in plots]

    title = Div(text = "<h2>Fitted measurements</h2>")
    doms += [title] + [observation_plots[s].make_doc() for s in simulation.sensors]

    c     = column(*doms)
    return Panel(child = c, title = "Retrieval", width = 600)

def make_document_factory(simulation):

    def make_document(doc):

        doc.title = "parts dashboard"

        sensor_p     = make_sensor_panel(simulation.sensors)
        atmosphere_p = make_atmosphere_panel(simulation, simulation.data_provider)
        retrieval_p  = make_retrieval_panel(simulation)



        t = Tabs(tabs = [sensor_p,
                         atmosphere_p,
                         retrieval_p])
        doc.add_root(t)

    return make_document

def dashboard(simulation):
    apps = {'/': Application(FunctionHandler(make_document_factory(simulation)))}
    standalone_server.start(apps, port = 8880)

