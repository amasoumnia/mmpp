import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mmpp.process import MarkovModulatedPoissonProcess

# Start Simulation
generator_matrix = np.array(
    [[-3, 2, 1], #low_demand
     [3, -5, 2], # med_demand
     [5, 3, -8]]  # surge_demand
)

intensities = np.array(
    [25 * 365, 100 * 365, 500 * 365]
)  # surge_demand (~25/day), med_demand (~100/day, surge_demand (~500/day)

mmpp = MarkovModulatedPoissonProcess(generator_matrix, intensities)

simulation_time = 3  # Simulate for across years
results = mmpp.simulate(start_intensity=25 * 365, end_time=simulation_time)

# Start Plotting
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.1,
    subplot_titles=("Cumulative Ticket Sales", "Underlying State Process"),
)

fig.add_trace(
    go.Scatter(
        x=results["Time"], y=results["Count"], mode="lines", name="Cumulative Sales"
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=results["Time"],
        y=results["CTMC State"] / 365,
        mode="lines",
        name="Intensity",
        line=dict(shape="hv"),
    ),
    row=2,
    col=1,
)

fig.update_layout(height=800, title_text="Airline Ticket Sales MMPP Simulation")
fig.update_xaxes(title_text="Time (years)")
fig.update_yaxes(title_text="Number of Tickets Sold", row=1, col=1)
fig.update_yaxes(title_text="Intensity (count / day)", row=2, col=1)

fig.add_annotation(
    x=0.03, y=500, text="Surge Demand", showarrow=False, xref="paper", yref="y2"
)
fig.add_annotation(
    x=0.03, y=100, text="Medium Demand", showarrow=False, xref="paper", yref="y2"
)
fig.add_annotation(
    x=0.03, y=25, text="Low Demand", showarrow=False, xref="paper", yref="y2"
)

fig.update_layout(
    {
        "paper_bgcolor": "rgba(0, 0, 0, 0)",
        "plot_bgcolor": "rgba(0, 0, 0, 0)",
    }
)

fig.write_image("mmpp_simulation.svg", scale=3, width=1000, height=750)