import plotly.graph_objects as go

def create_empty_3d_figure():
    """
    Creates an empty Plotly 3D figure with axes and styling.
    """
    fig = go.Figure()

    # Add a dummy trace so axes are visible even if empty
    # We can use a single point at (0,0,0) or just set the range
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
