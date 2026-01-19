import plotly.graph_objects as go
from typing import List, Tuple, Any
import shapely.geometry as sg
from design_params import Mesh3D

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

def plot_pit_data(
    up_string: List[Tuple[float, float, float]],
    highlight_index: int = None,
    benches: List[Any] = None
):
    """
    Plots the Ultimate Pit string and optionally highlights a specific point.
    Also plots generated benches if provided.
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

    # Plot Benches
    if benches:
        for bench in benches:
            # 1. Plot Mesh Faces (Walls)
            if bench.face_mesh and bench.face_mesh.vertices:
                v = bench.face_mesh.vertices
                f = bench.face_mesh.faces

                x_v = [p[0] for p in v]
                y_v = [p[1] for p in v]
                z_v = [p[2] for p in v]

                i_f = [t[0] for t in f]
                j_f = [t[1] for t in f]
                k_f = [t[2] for t in f]

                fig.add_trace(go.Mesh3d(
                    x=x_v, y=y_v, z=z_v,
                    i=i_f, j=j_f, k=k_f,
                    color='gray',
                    opacity=0.8,
                    name=f'Bench {bench.bench_id} Face',
                    showlegend=True,
                    hoverinfo='skip',
                    flatshading=True,
                    lighting=dict(ambient=0.5, diffuse=0.5)
                ))

            # 2. Plot Mesh Berms (Floors)
            if bench.berm_mesh and bench.berm_mesh.vertices:
                v = bench.berm_mesh.vertices
                f = bench.berm_mesh.faces

                x_v = [p[0] for p in v]
                y_v = [p[1] for p in v]
                z_v = [p[2] for p in v]

                i_f = [t[0] for t in f]
                j_f = [t[1] for t in f]
                k_f = [t[2] for t in f]

                fig.add_trace(go.Mesh3d(
                    x=x_v, y=y_v, z=z_v,
                    i=i_f, j=j_f, k=k_f,
                    color='lightgray',
                    opacity=0.6,
                    name=f'Bench {bench.bench_id} Berm',
                    showlegend=True,
                    hoverinfo='skip',
                    flatshading=True
                ))

            # 3. Plot Crest Lines (Wireframe)
            if bench.crest_polys:
                for poly in bench.crest_polys:
                    if not poly.is_empty:
                        c_x, c_y = poly.exterior.xy
                        c_z = [bench.z_crest] * len(c_x)

                        fig.add_trace(go.Scatter3d(
                            x=list(c_x), y=list(c_y), z=c_z,
                            mode='lines',
                            line=dict(color='green', width=3),
                            name=f'Bench {bench.bench_id} Crest',
                            showlegend=False
                        ))

            # 4. Plot Toe Lines (Wireframe)
            if bench.toe_polys:
                for poly in bench.toe_polys:
                    if not poly.is_empty:
                        t_x, t_y = poly.exterior.xy
                        t_z = [bench.z_toe] * len(t_x)

                        fig.add_trace(go.Scatter3d(
                            x=list(t_x), y=list(t_y), z=t_z,
                            mode='lines',
                            line=dict(color='orange', width=3),
                            name=f'Bench {bench.bench_id} Toe',
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
