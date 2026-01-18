import streamlit as st
import pit_viz as viz
import data_loader
import pit_design

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

batter_angle = st.sidebar.number_input(
    "Batter Angle (deg)",
    min_value=10.0,
    max_value=89.0,
    value=75.0,
    step=1.0,
    help="Face angle (from horizontal)"
)

berm_width = st.sidebar.number_input(
    "Berm Width (m)",
    min_value=0.0,
    max_value=50.0,
    value=5.0,
    step=0.5,
    help="Width of the catch berm"
)

target_elev = st.sidebar.number_input(
    "Target Elevation (m)",
    value=0.0,
    step=10.0,
    help="Bottom elevation to reach"
)

# Generate Design Button
if st.sidebar.button("Generate Pit Design"):
    params = pit_design.PitDesignParams(
        bench_height=bench_height,
        batter_angle_deg=batter_angle,
        berm_width=berm_width,
        target_elevation=target_elev
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

    if st.session_state['diagnostics']:
        st.write("---")
        st.write("Generation Diagnostics:")
        st.json(st.session_state['diagnostics'])

    if st.session_state['benches']:
        st.write(f"Generated {len(st.session_state['benches'])} benches")
