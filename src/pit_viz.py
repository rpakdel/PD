import plotly.graph_objects as go
from typing import List, Tuple

def create_empty_3d_figure():
    """
    Creates an empty Plotly 3D figure with axes and styling.
    """
    fig = go.Figure()

    # Add a dummy trace so axes are visible even if empty
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=0, opacity=0),
        name='Origin',
        showlegend=False
    ))

    fig.update_layout(
        title="Open Pit Design 3D Viewer",
        scene=dict(
            xaxis=dict(title='X (Easting)'),
            yaxis=dict(title='Y (Northing)'),
            zaxis=dict(title='Z (Elevation)'),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=600
    )

    return fig

def plot_pit_data(up_string: List[Tuple[float, float, float]], highlight_index: int = None):
    """
    Plots the Ultimate Pit string and optionally highlights a specific point.
    """
    fig = go.Figure()

    # Unpack coordinates
    if up_string:
        xs = [p[0] for p in up_string]
        ys = [p[1] for p in up_string]
        zs = [p[2] for p in up_string]

        # Plot UP string line
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='lines+markers',
            line=dict(color='blue', width=4),
            marker=dict(size=3, color='blue'),
            name='UP String'
        ))

        # Highlight specific index
        if highlight_index is not None and 0 <= highlight_index < len(up_string):
            hx, hy, hz = up_string[highlight_index]
            fig.add_trace(go.Scatter3d(
                x=[hx], y=[hy], z=[hz],
                mode='markers+text',
                marker=dict(size=10, color='red', symbol='circle'),
                text=[str(highlight_index)],
                textposition="top center",
                name='Selected Point'
            ))

    fig.update_layout(
        title="Open Pit Design 3D Viewer",
        scene=dict(
            xaxis=dict(title='X (Easting)'),
            yaxis=dict(title='Y (Northing)'),
            zaxis=dict(title='Z (Elevation)'),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=600
    )

    return fig
