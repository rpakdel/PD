import streamlit as st
import pandas as pd
import pit_viz as viz
import data_loader
import pit_design
import design_params
import ramp_design

st.set_page_config(
    page_title="Open Pit Design PoC",
    page_icon="⛏️",
    layout="wide"
)

st.title("Open Pit Design Generator (PoC)")
st.markdown("""
This tool generates a pit design from an Ultimate Pit (UP) string.
""")

# Load UP string
up_string = data_loader.get_sample_up_string()

# Initialize Session State
if 'benches' not in st.session_state:
    st.session_state['benches'] = []
if 'diagnostics' not in st.session_state:
    st.session_state['diagnostics'] = {}
if 'ramp_slices' not in st.session_state:
    st.session_state['ramp_slices'] = []
if 'ramp_centerline' not in st.session_state:
    st.session_state['ramp_centerline'] = []
if 'ramp_corridor' not in st.session_state:
    st.session_state['ramp_corridor'] = None

# Sidebar for inputs
st.sidebar.header("Design Parameters")

bench_height = st.sidebar.number_input(
    "Bench Height (m)",
    min_value=1.0,
    max_value=50.0,
    value=10.0,
    step=1.0,
    help="Vertical height of each bench"
)

# Checkbox for variable params (Moved up to control visibility/state of other inputs)
use_variable_params = st.sidebar.checkbox("Use Elevation-based Parameters", value=False)

# Uniform parameters - disabled if variable params are used
batter_angle = st.sidebar.number_input(
    "Batter Angle (deg)",
    min_value=10.0,
    max_value=89.0,
    value=75.0,
    step=1.0,
    help="Face angle (from horizontal)",
    disabled=use_variable_params
)

berm_width = st.sidebar.number_input(
    "Berm Width (m)",
    min_value=0.0,
    max_value=50.0,
    value=5.0,
    step=0.5,
    help="Width of the catch berm",
    disabled=use_variable_params
)

design_direction = st.sidebar.radio(
    "Generation Direction",
    options=["Downward", "Upward"],
    index=0,
    help="Direction to generate the pit. 'Downward' projects the string down. 'Upward' projects the string up."
)

target_elev_label = "Target Elevation (m)"
target_elev_help = "Target elevation to reach"
default_target = 0.0

if design_direction == "Downward":
    target_elev_help = "Bottom elevation to reach (e.g. 0 or -100)"
    default_target = 0.0
else:
    target_elev_help = "Top elevation to reach (e.g. 200)"
    default_target = 200.0

target_elev = st.sidebar.number_input(
    target_elev_label,
    value=default_target,
    step=10.0,
    help=target_elev_help
)

st.sidebar.markdown("---")
variable_params_list = []
parsing_error = False

if use_variable_params:
    st.sidebar.markdown("Define parameters for specific elevation ranges. Ranges are inclusive.")

    # Default data for the editor
    default_data = {
        "Min Elev": [0, 100.0, 125.0, 150.0, 175.0],
        "Max Elev": [100, 125.0, 150.0, 175.0, 200.0],
        "Angle (deg)": [70, 75.0, 80.0, 85.0, 90.0],
        "Width (m)": [8, 10.0, 15.0, 20.0, 5.0]
    }

    edited_df = st.sidebar.data_editor(
        pd.DataFrame(default_data),
        num_rows="dynamic",
        use_container_width=True
    )

    # Parse DataFrame to List[DesignBlock]
    if not edited_df.empty:
        for _, row in edited_df.iterrows():
            try:
                block = design_params.DesignBlock(
                    z_start=float(row["Min Elev"]),
                    z_end=float(row["Max Elev"]),
                    batter_angle_deg=float(row["Angle (deg)"]),
                    berm_width=float(row["Width (m)"])
                )
                variable_params_list.append(block)
            except (ValueError, KeyError):
                st.sidebar.error("Invalid data in parameter table.")
                parsing_error = True

# Generate Design Button
if st.sidebar.button("Generate Pit Design"):
    if parsing_error:
        st.error("Please fix errors in the parameter table before generating.")
    elif use_variable_params and not variable_params_list:
        st.error("Use Elevation-based Parameters is checked, but no valid ranges are defined.")
    else:
        # Explicitly cast arguments to ensure types are correct, avoiding potential TypeError
        params = design_params.PitDesignParams(
            bench_height=float(bench_height),
            batter_angle_deg=float(batter_angle),
            berm_width=float(berm_width),
            target_elevation=float(target_elev),
            design_direction=str(design_direction),
            variable_params=variable_params_list
        )
        # Store params for later use (e.g. ramp design)
        st.session_state['design_params'] = params

        benches, diagnostics = pit_design.generate_pit_benches(up_string, params)
        st.session_state['benches'] = benches
        st.session_state['diagnostics'] = diagnostics
        st.session_state['ramp_slices'] = [] # Reset ramp slices
        st.session_state['ramp_centerline'] = []
        st.session_state['ramp_corridor'] = None


st.sidebar.markdown("---")
st.sidebar.header("Ramp Design (Beta)")

# Ramp Inputs
ramp_width = st.sidebar.number_input("Ramp Width (m)", value=20.0, step=1.0)
ramp_grade = st.sidebar.number_input("Max Grade (%)", value=10.0, step=0.5)
ramp_z_step = st.sidebar.number_input("Solver Z Step (m)", value=5.0, step=1.0, help="Vertical step size for ramp slices")
ramp_mode = st.sidebar.selectbox("Ramp Mode", options=["spiral", "switchback"], index=0)

# Start Point Selection
# We default to using the 'Highlighted Point' from below as the start point x,y
# And top Z.
start_pt_idx = 0

if st.sidebar.button("Preview Ramp Slices"):
    if not st.session_state['benches']:
        st.error("Please generate a pit design first.")
    elif 'design_params' not in st.session_state:
         st.error("Design parameters missing. Please regenerate the pit.")
    else:
        ramp_params = design_params.RampParams(
            ramp_width=float(ramp_width),
            grade_max=float(ramp_grade)/100.0,
            z_step=float(ramp_z_step),
            mode=ramp_mode
        )
        st.session_state['ramp_params'] = ramp_params

        slices = ramp_design.create_slices(
            st.session_state['benches'],
            ramp_params,
            st.session_state['design_params']
        )
        st.session_state['ramp_slices'] = slices
        st.success(f"Generated {len(slices)} slices for ramp preview.")

if st.sidebar.button("Generate Ramp"):
    if not st.session_state['ramp_slices']:
         st.error("Please preview ramp slices first to initialize data.")
    else:
        # Get Start Point
        # Use highlighted point index or defaults
        # Assuming we want to start at the top of the pit near a specific point
        # Let's pick the point from UP string

        # We need to know which index is selected below
        # Accessing widget state via key might be better, but we are using flow
        # Let's rely on st.session_state variable if we key it
        pass # We will get it from the widget below if possible, or defaulting to 0

        # Actually, let's just grab the one from UP string at index 0 for now or add a selector here
        # But we have one below.

        # To make it robust:
        # We need x,y from UP string. Z from top slice.

        # Use the slider value if available? Streamlit reruns script, so 'selected_index' variable below is available?
        # No, execution is top-down. We need to move the slider up or use session state.

        # Let's just default to index 0 for now.
        start_pt_uv = up_string[0]
        # We can add a "Start Point Index" input here specifically for ramp

        ramp_params = st.session_state.get('ramp_params')
        if not ramp_params:
             ramp_params = design_params.RampParams(
                ramp_width=float(ramp_width),
                grade_max=float(ramp_grade)/100.0,
                z_step=float(ramp_z_step),
                mode=ramp_mode
            )

        slices = st.session_state['ramp_slices']
        target_z = float(target_elev) # Same as pit target

        start_point_xy = (start_pt_uv[0], start_pt_uv[1])

        path, diag = ramp_design.solve_ramp(slices, start_point_xy, target_z, ramp_params)

        if path:
            st.session_state['ramp_centerline'] = path
            st.success("Ramp generated successfully!")

            # Generate Corridor
            left, right = ramp_design.generate_ramp_corridor(path, ramp_params.ramp_width)
            st.session_state['ramp_corridor'] = (left, right)

        else:
            st.error("Ramp generation failed.")
            st.json(diag)


st.sidebar.markdown("---")
st.sidebar.header("UP String Inspector")

# Index selector
num_points = len(up_string)
unique_point_count = num_points - 1 if num_points > 1 and up_string[0] == up_string[-1] else num_points

selected_index = st.sidebar.number_input(
    "Highlight Point Index",
    min_value=0,
    max_value=unique_point_count - 1 if unique_point_count > 0 else 0,
    value=0,
    step=1
)

# Display coordinates of selected point
if unique_point_count > 0:
    sel_pt = up_string[selected_index]
    st.sidebar.markdown(f"**Selected Point:**")
    st.sidebar.code(f"X: {sel_pt[0]:.2f}\nY: {sel_pt[1]:.2f}\nZ: {sel_pt[2]:.2f}")
else:
    st.sidebar.warning("No points in UP string")


st.sidebar.markdown("---")
st.sidebar.header("Data Import (Placeholder)")
st.sidebar.info("DXF Topography and UP String import will be implemented in future phases.")

# Main area visualization
st.subheader("3D Visualization")

# Check for errors in diagnostics and show them prominently
if st.session_state.get('diagnostics') and "error" in st.session_state['diagnostics']:
    st.error(f"Generation Error: {st.session_state['diagnostics']['error']}")

# Create and display the plot
extra_traces = []
if st.session_state['ramp_slices']:
    import plotly.graph_objects as go

    # Visualize Free Poly (Green)
    step_vis = max(1, len(st.session_state['ramp_slices']) // 10)

    for i, sl in enumerate(st.session_state['ramp_slices']):
        if i % step_vis != 0 and i != 0 and i != len(st.session_state['ramp_slices'])-1:
            continue

        z = sl.z

        # Free Poly
        if not sl.free_poly.is_empty:
            if sl.free_poly.geom_type == 'Polygon':
                polys = [sl.free_poly]
            else:
                polys = list(sl.free_poly.geoms)

            for poly in polys:
                x, y = poly.exterior.xy
                z_arr = [z] * len(x)
                extra_traces.append(go.Scatter3d(
                    x=list(x), y=list(y), z=z_arr,
                    mode='lines',
                    line=dict(color='cyan', width=2),
                    name=f'Free Poly {z:.1f}'
                ))

fig = viz.plot_pit_data(
    up_string,
    highlight_index=selected_index,
    benches=st.session_state['benches'],
    ramp_centerline=st.session_state.get('ramp_centerline'),
    ramp_corridor=st.session_state.get('ramp_corridor')
)
if extra_traces:
    fig.add_traces(extra_traces)

st.plotly_chart(fig, use_container_width=True)

# Debug/Diagnostics section
with st.expander("Diagnostics & Data"):
    st.write(f"Bench Height: {bench_height} m")
    st.write(f"Batter Angle: {batter_angle} deg")
    st.write(f"Berm Width: {berm_width} m")
    st.write(f"Target Elevation: {target_elev} m")
    st.write(f"UP String Points: {len(up_string)}")

    if st.session_state.get('diagnostics'):
        st.write("---")
        st.write("Generation Diagnostics:")
        diag = st.session_state['diagnostics']
        if "error" in diag:
            st.error(f"Generation Error: {diag['error']}")
        st.json(diag)

    if st.session_state.get('benches'):
        st.write(f"Generated {len(st.session_state['benches'])} benches")

    if st.session_state.get('ramp_slices'):
        st.write(f"Generated {len(st.session_state['ramp_slices'])} ramp slices")
