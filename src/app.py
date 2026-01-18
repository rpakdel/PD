import streamlit as st
import viz

st.set_page_config(
    page_title="Open Pit Design PoC",
    page_icon="⛏️",
    layout="wide"
)

st.title("Open Pit Design Generator (PoC)")
st.markdown("""
This tool generates a pit design from an Ultimate Pit (UP) string.
""")

# Sidebar for inputs
st.sidebar.header("Design Parameters")

# Dummy inputs based on Agents.md
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

# Placeholder for future inputs
st.sidebar.markdown("---")
st.sidebar.header("Data Import (Placeholder)")
st.sidebar.info("DXF Topography and UP String import will be implemented in future phases.")

# Main area visualization
st.subheader("3D Visualization")

# Create and display the plot
fig = viz.create_empty_3d_figure()
st.plotly_chart(fig)

# Debug/Diagnostics section
with st.expander("Diagnostics & Data"):
    st.write(f"Bench Height: {bench_height} m")
    st.write(f"Batter Angle: {batter_angle} deg")
    st.write(f"Berm Width: {berm_width} m")
    st.write(f"Target Elevation: {target_elev} m")
