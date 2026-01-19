import streamlit as st
import pandas as pd
import pit_viz as viz
import data_loader
import pit_design
import design_params

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

        benches, diagnostics = pit_design.generate_pit_benches(up_string, params)
        st.session_state['benches'] = benches
        st.session_state['diagnostics'] = diagnostics
else:
    if 'benches' not in st.session_state:
        st.session_state['benches'] = []
        st.session_state['diagnostics'] = {}

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
fig = viz.plot_pit_data(
    up_string,
    highlight_index=selected_index,
    benches=st.session_state['benches']
)
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
